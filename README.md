# HALO

Heat-Aware Layout Optimizer — Capstone 2026

## Formatter & Linter

### Python (backend, engine)

[Ruff](https://docs.astral.sh/ruff/)를 사용합니다. 설정 파일은 프로젝트 루트의 `ruff.toml`입니다.

```bash
# 설치
pip install ruff
# 또는
conda install -c conda-forge ruff

# 린트
ruff check .

# 린트 + 자동 수정
ruff check --fix .

# 포맷팅
ruff format .
```

VS Code 사용 시 Ruff 확장을 설치하면 저장할 때 자동으로 포맷 및 import 정리가 적용됩니다.

### Frontend

[Biome](https://biomejs.dev/)를 사용합니다. 설정 파일은 `frontend/biome.json`입니다.

```bash
cd frontend

# 린트
pnpm lint

# 린트 + 자동 수정
pnpm lint:fix

# 포맷팅
pnpm format
```
