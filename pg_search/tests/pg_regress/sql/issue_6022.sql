-- Regression test for score projection when a BM25 Top-K query is filtered
-- through a related table.  PostgreSQL can represent the filter as a join,
-- semi-join, or retained SubPlan; every shape must consume the score produced
-- by ParadeDB rather than evaluate the placeholder function itself.

CREATE EXTENSION IF NOT EXISTS pg_search;

SET max_parallel_workers_per_gather = 0;
SET paradedb.enable_join_custom_scan = off;

DROP TABLE IF EXISTS issue_6022_documents CASCADE;
DROP TABLE IF EXISTS issue_6022_access CASCADE;

CREATE TABLE issue_6022_documents (
    id INTEGER PRIMARY KEY,
    body TEXT NOT NULL
);

CREATE TABLE issue_6022_access (
    document_id INTEGER PRIMARY KEY,
    audience TEXT NOT NULL
);

INSERT INTO issue_6022_documents (id, body)
SELECT id,
       repeat('alpha ', alpha_count) || repeat('context ', 50 - alpha_count)
FROM (
    SELECT id,
           CASE id
               WHEN 3 THEN 40
               WHEN 6 THEN 30
               WHEN 1 THEN 25
               WHEN 2 THEN 20
               WHEN 10 THEN 10
               ELSE 1
           END AS alpha_count
    FROM generate_series(1, 12) AS id
) AS documents;

INSERT INTO issue_6022_access (document_id, audience)
SELECT id,
       CASE
           WHEN id IN (2, 6, 10) THEN 'selective'
           WHEN id <= 10 THEN 'broad'
           ELSE 'excluded'
       END
FROM generate_series(1, 12) AS id;

CREATE INDEX issue_6022_documents_bm25 ON issue_6022_documents
USING paradedb (id, body)
WITH (key_field = 'id');

ANALYZE issue_6022_documents;
ANALYZE issue_6022_access;

-- Inspect EXPLAIN through a helper so the regression assertion is stable across
-- PostgreSQL versions while still checking the important planner responsibilities.
CREATE FUNCTION issue_6022_explain()
RETURNS TABLE(base_scan BOOLEAN, scores BOOLEAN, native_join BOOLEAN, native_sort BOOLEAN)
LANGUAGE plpgsql
AS $$
DECLARE
    line TEXT;
BEGIN
    base_scan := false;
    scores := false;
    native_join := false;
    native_sort := false;

    FOR line IN EXECUTE $query$
        EXPLAIN (COSTS OFF)
        SELECT d.id, round(paradedb.score(d.id)::numeric, 4) AS score
        FROM issue_6022_documents AS d
        JOIN (
            SELECT document_id
            FROM issue_6022_access
            WHERE audience = 'selective'
        ) AS allowed ON allowed.document_id = d.id
        WHERE d.body @@@ 'alpha'
        ORDER BY paradedb.score(d.id) DESC, d.id DESC
        LIMIT 3
    $query$
    LOOP
        base_scan := base_scan OR line LIKE '%Custom Scan (ParadeDB Base Scan)%';
        scores := scores OR line LIKE '%Scores: true%';
        native_join := native_join OR line LIKE '%Join%' OR line LIKE '%Nested Loop%';
        native_sort := native_sort OR line LIKE '%Sort%';
    END LOOP;

    RETURN NEXT;
END;
$$;

SELECT * FROM issue_6022_explain();

-- Selective predicate: all equivalent filter forms must return the same Top-K.
SELECT 'join' AS shape, d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
JOIN (
    SELECT document_id FROM issue_6022_access WHERE audience = 'selective'
) AS allowed ON allowed.document_id = d.id
WHERE d.body @@@ 'alpha'
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

SELECT 'in' AS shape, d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
WHERE d.body @@@ 'alpha'
  AND d.id IN (
      SELECT document_id FROM issue_6022_access WHERE audience = 'selective'
  )
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

SELECT 'in-limit-all' AS shape, d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
WHERE d.body @@@ 'alpha'
  AND d.id IN (
      SELECT document_id
      FROM issue_6022_access
      WHERE audience = 'selective'
      LIMIT ALL
  )
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

SELECT 'exists' AS shape, d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
WHERE d.body @@@ 'alpha'
  AND EXISTS (
      SELECT 1
      FROM issue_6022_access AS allowed
      WHERE allowed.document_id = d.id
        AND allowed.audience = 'selective'
  )
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

-- Broader statistics/cardinality can produce a different native plan shape.
SELECT 'join-broad' AS shape, d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
JOIN (
    SELECT document_id FROM issue_6022_access WHERE audience <> 'excluded'
) AS allowed ON allowed.document_id = d.id
WHERE d.body @@@ 'alpha'
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

SELECT 'in-broad' AS shape, d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
WHERE d.body @@@ 'alpha'
  AND d.id IN (
      SELECT document_id FROM issue_6022_access WHERE audience <> 'excluded'
  )
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

SELECT 'in-limit-all-broad' AS shape, d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
WHERE d.body @@@ 'alpha'
  AND d.id IN (
      SELECT document_id
      FROM issue_6022_access
      WHERE audience <> 'excluded'
      LIMIT ALL
  )
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

SELECT 'exists-broad' AS shape, d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
WHERE d.body @@@ 'alpha'
  AND EXISTS (
      SELECT 1
      FROM issue_6022_access AS allowed
      WHERE allowed.document_id = d.id
        AND allowed.audience <> 'excluded'
  )
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

-- Empty related result and an unfiltered Top-K control.
SELECT d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
WHERE d.body @@@ 'alpha'
  AND d.id IN (
      SELECT document_id FROM issue_6022_access WHERE audience = 'missing'
  )
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

SELECT d.id, paradedb.score(d.id) > 0 AS scored
FROM issue_6022_documents AS d
WHERE d.body @@@ 'alpha'
ORDER BY paradedb.score(d.id) DESC, d.id DESC
LIMIT 3;

DROP TABLE issue_6022_access;
DROP TABLE issue_6022_documents CASCADE;
DROP FUNCTION issue_6022_explain();

RESET paradedb.enable_join_custom_scan;
RESET max_parallel_workers_per_gather;
