# StateScope v0.1.2 source release readiness

**SOURCE_RELEASE_READY** for the reviewed software source. The related research evidence is retained for local review. Evidence-archive redistribution is not approved.

| Check | Result | Evidence |
|---|---|---|
| Software attribution and license | PASS | SIQI LI, copyright 2026, Apache-2.0. LICENSE is byte-identical to the Apache Software Foundation text. NOTICE records the verified holder. |
| Citation metadata | PASS | The official CFF 1.2.0 schema accepts CITATION.cff. Its author, ORCID, software version and license agree with the confirmed metadata. |
| Python distributions | PASS | The sdist and the wheel rebuilt from it contain LICENSE and NOTICE. The wheel metadata records Apache-2.0. All 118 runtime files match the original research wheel byte for byte. The rebuilt wheel installed offline in a clean environment and passed pip check. |
| Repository-only quick reproduction | PASS | All four synthetic cases and the result verifier passed with the reviewed wheel on Python 3.13.9 and scikit-learn 1.8.0. The command took 4.34 s in the reviewed-source directory. The archived checks were explicitly reported as NOT RUN. |
| Regression suite | PASS | All 13 tests passed. The test runner reported 3.834 s. |
| Authorized archival verification | PASS | Table 8 and Table 16 recomputation and evidence comparisons passed in a separate local run. The command took 11.74 s. A supplied but missing archive returned exit code 1 and NOT AVAILABLE for the required archive cases. |
| Graph replay | PASS | All 660 records exactly match the archived records. The 260 structured and 400 generated records contain 640 checks without the wrong-version injection and 20 deliberate injections. The latter produced 20 lower-bound breaches and five false exposures. The remaining 640 produced zero false determinate results. |
| Public source contents | PASS | The software bundle excludes private experiment records, third-party snapshots, saved model arrays, dependency caches, and local verification logs. The source candidate passed the credential and private-path checks. |
| Original research materials | PASS | The original wheel, original source candidate ZIP, evidence ZIP, 1,507 manifested evidence files, and the 32-page Revised PDF match their prior checksums. |

The dependency setup used a local wheelhouse. Creating the virtual environment, bootstrapping pip, installing dependencies and tools, and installing the reviewed wheel took 0.14 s, 11.54 s, 91.11 s, and 2.91 s, respectively. The first sdist unpack attempt hit a Windows path-permission error. The repeat extracted the same sdist to a fresh staging path, validated its member names, and built the wheel there.

The manuscript's Evaluation Overview currently reads “660 valid and 20 invalid edge checks.” The separate manuscript correction clarifies that the 20 injections are included in the total of 660. The paper and original research records are retained. GitHub URL, Zenodo DOI and release date can be recorded when those identifiers exist. They are omitted from the public citation file. This review covers the software bundle. Research data and original derived evidence remain under their existing distribution status.

