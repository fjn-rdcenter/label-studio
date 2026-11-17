# syntax=docker/dockerfile:1
ARG NODE_VERSION=18

################################ Stage: frontend-builder (build frontend assets)
FROM --platform=${BUILDPLATFORM} node:${NODE_VERSION} AS frontend-builder
ENV BUILD_NO_SERVER=true \
    BUILD_NO_HASH=true \
    BUILD_NO_CHUNKS=true \
    BUILD_MODULE=true \
    YARN_CACHE_FOLDER=/root/web/.yarn \
    NODE_ENV=production
ENV http_proxy=http://hoc-nt:Abc1234567@proxy.fujinet.vn:8080
ENV https_proxy=http://hoc-nt:Abc1234567@proxy.fujinet.vn:8080
ENV no_proxy=localhost
ENV HTTP_PROXY=http://hoc-nt:Abc1234567@proxy.fujinet.vn:8080
ENV HTTPS_PROXY=http://hoc-nt:Abc1234567@proxy.fujinet.vn:8080
ENV NO_PROXY=localhost

WORKDIR /label-studio/web

COPY ./certs/FJS_SSL_Certificates.crt /usr/local/share/ca-certificates/FJS_SSL_Certificates.crt
RUN if ! grep -q -- '-----BEGIN CERTIFICATE-----' /usr/local/share/ca-certificates/FJS_SSL_Certificates.crt; then \
      openssl x509 -inform DER \
        -in /usr/local/share/ca-certificates/FJS_SSL_Certificates.crt \
        -outform PEM \
        -out /tmp/FJS_SSL_Certificates.pem \
      && mv /tmp/FJS_SSL_Certificates.pem /usr/local/share/ca-certificates/FJS_SSL_Certificates.crt; \
    fi \
    && update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# Node ships its own CA list and ignores the system store unless told otherwise.
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

# Fix Docker Arm64 Build
RUN yarn config set registry https://registry.npmjs.org/
RUN yarn config set network-timeout 1200000 # HTTP timeout used when downloading packages, set to 20 minutes

COPY web/package.json .
COPY web/yarn.lock .
COPY web/tools tools
RUN --mount=type=cache,target=${YARN_CACHE_FOLDER},sharing=locked \
    yarn install --prefer-offline --no-progress --pure-lockfile --frozen-lockfile --ignore-engines --non-interactive --production=false

COPY web .
COPY pyproject.toml ../pyproject.toml
RUN --mount=type=cache,target=${YARN_CACHE_FOLDER},sharing=locked \
    --mount=type=bind,source=.git,target=../.git \
    yarn run build && yarn version:libs

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LS_DIR=/label-studio \
    PIP_CACHE_DIR=$HOME/.cache \
    POETRY_CACHE_DIR=$HOME/.poetry-cache \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HTTP_TIMEOUT=300 \
    POETRY_INSTALLER_MAX_WORKERS=4 \
    DJANGO_SETTINGS_MODULE=core.settings.label_studio \
    LABEL_STUDIO_BASE_DATA_DIR=/label-studio/data \
    OPT_DIR=/opt/heartex/instance-data/etc \
    SETUPTOOLS_USE_DISTUTILS=stdlib

WORKDIR $LS_DIR

# install packages
# RUN set -eux \
#  && apt-get update \
#  && apt-get install --no-install-recommends --no-install-suggests -y \
#     build-essential postgresql-client libmysqlclient-dev mysql-client python3-pip python3-dev \
#     git libxml2-dev libxslt-dev zlib1g-dev gnupg curl lsb-release libpq-dev dnsutils vim && \
#     apt-get purge --assume-yes --auto-remove --option APT::AutoRemove::RecommendsImportant=false \
#      --option APT::AutoRemove::SuggestsImportant=false && rm -rf /var/lib/apt/lists/* /tmp/*


RUN set -eux \
    && apt-get update \
    && apt-get install --no-install-recommends --no-install-suggests -y \
        build-essential \
        postgresql-client \
        libmysqlclient-dev \
        mysql-client \
        python3-pip \
        python3-dev \
        git \
        libxml2-dev \
        libxslt-dev \
        zlib1g-dev \
        gnupg \
        curl \
        lsb-release \
        libpq-dev \
        dnsutils \
        vim \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Trust the corporate proxy CA here too (already converted to PEM in frontend-builder).
COPY --from=frontend-builder /usr/local/share/ca-certificates/FJS_SSL_Certificates.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

RUN --mount=type=cache,target=$PIP_CACHE_DIR,uid=1001,gid=0 \
    pip3 install --upgrade pip setuptools && \
    pip3 install \
      "poetry==1.6.1" \
      "packaging==23.2" \
      "virtualenv==20.24.6" \
      uwsgi uwsgitop


# incapsulate nginx install & configure to a single layer
RUN set -eux; \
    curl -sSL https://nginx.org/keys/nginx_signing.key | apt-key add - && \
    echo "deb https://nginx.org/packages/mainline/ubuntu/ $(lsb_release -cs) nginx" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y nginx && \
    apt-get purge --assume-yes --auto-remove --option APT::AutoRemove::RecommendsImportant=false \
     --option APT::AutoRemove::SuggestsImportant=false && rm -rf /var/lib/apt/lists/* /tmp/* && \
    nginx -v

COPY --chown=1001:0 deploy/default.conf /etc/nginx/nginx.conf

RUN set -eux; \
    mkdir -p $OPT_DIR /var/log/nginx /var/cache/nginx /etc/nginx && \
    chown -R 1001:0 $OPT_DIR /var/log/nginx /var/cache/nginx /etc/nginx

# Copy essential files for installing Label Studio and its dependencies
COPY --chown=1001:0 pyproject.toml .
COPY --chown=1001:0 poetry.lock .
COPY --chown=1001:0 README.md .
COPY --chown=1001:0 label_studio/__init__.py ./label_studio/__init__.py

# Install all dependencies AND the label_studio package into system Python.
# Installing WITH root creates the label-studio dist-info metadata required by
# importlib.metadata.metadata('label-studio') in label_studio/__init__.py.
# PYTHONPATH=/label-studio ensures Python uses the full copied source directory.
#
# Poetry shares the system environment with the application, so keep its
# packaging dependencies aligned with poetry.lock to avoid replacing them
# while Poetry is running. Test dependencies are not needed in this image.
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry check --lock && poetry install

COPY --chown=1001:0 LICENSE LICENSE
COPY --chown=1001:0 licenses licenses
COPY --chown=1001:0 label_studio label_studio
COPY --chown=1001:0 deploy deploy

COPY --chown=1001:0 --from=frontend-builder /label-studio/web/dist $LS_DIR/web/dist


RUN python3 label_studio/manage.py collectstatic --no-input && \
    chown -R 1001:0 $LS_DIR && \
    chmod -R g=u $LS_DIR

ENV HOME=$LS_DIR

EXPOSE 8080

USER 1001

ENTRYPOINT ["./deploy/docker-entrypoint.sh"]
CMD ["label-studio"]
