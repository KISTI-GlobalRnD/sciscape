# SciScape

이 저장소는 `sciscape` Python 패키지(Leiden 클러스터링 + 키워드 추출)를 포함합니다.

- 패키지 문서: `sciscape/README.md`
- 주요 import:
  - `sciscape.clustering`
  - `sciscape.keyword_extraction`

호환을 위해 기존 `sos.*` import도 shim으로 유지합니다.

## Scope

이 저장소는 네트워크 기반 클러스터링/키워드 추출 파이프라인(`sciscape`)만 포함합니다.
보고서 생성(문서 머지/변환/웹 리포트 등) 파이프라인은 별도 프로젝트에서 관리합니다.
