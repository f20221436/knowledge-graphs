// v2 additions to the schema (v1 constraints are created by the v1 loader). Safe to re-run.
CREATE CONSTRAINT behaviour_state_code IF NOT EXISTS FOR (s:BehaviourState) REQUIRE s.code IS UNIQUE;
CREATE CONSTRAINT motif_pattern IF NOT EXISTS FOR (p:Motif) REQUIRE p.pattern IS UNIQUE;
