# Base Image
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# 필수 도구 설치 (curl 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv 설치 (astral 공식 설치 스크립트, 시스템 전역 경로에 설치)
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 경로 정의
WORKDIR /workdir

# pyproject.toml, uv.lock, README.md 먼저 복사하여 캐시 최적화
COPY pyproject.toml uv.lock README.md /workdir/

# uv로 의존성 설치 (시스템 환경에 설치)
RUN uv sync --no-dev --frozen

# 로컬에 있는 소스코드를 컨테이너로 복사
COPY . /workdir

# Python 경로 설정
# ENV PYTHONPATH=/usr/local/bin/python3.12

# Poetry 바이너리 권한 확인 및 설정
# RUN chmod +x /usr/local/bin/poetry

# # Poetry가 설치된 Python을 사용하도록 설정
# RUN sed -i '1s|^.*$|#!/usr/local/bin/python3.12|' /usr/local/bin/poetry

# # 권한과 바이너리 위치 확인
# RUN ls -l /usr/local/bin/poetry

# Port Expose
EXPOSE 18000

# command to run
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "18000"]