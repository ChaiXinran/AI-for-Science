#!/usr/bin/env bash

resolve_dataset_dir() {
    local root="$1"
    local preferred="$2"
    local fallback_pattern="$3"

    if [[ ! -d "${root}" ]]; then
        echo "ERROR: DATA_ROOT does not exist: ${root}" >&2
        echo "Existing parent directories:" >&2
        ls -lah "$(dirname "${root}")" >&2 || true
        return 2
    fi

    if [[ -d "${root}/${preferred}" ]]; then
        printf '%s\n' "${root}/${preferred}"
        return 0
    fi

    local found
    found="$(find "${root}" -maxdepth 3 -type d -name "${preferred}" -print -quit 2>/dev/null || true)"
    if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}"
        return 0
    fi

    found="$(find "${root}" -maxdepth 3 -type d -iname "${fallback_pattern}" -print -quit 2>/dev/null || true)"
    if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}"
        return 0
    fi

    echo "ERROR: could not find ${preferred} under ${root}" >&2
    echo "Top directories under DATA_ROOT:" >&2
    find "${root}" -maxdepth 3 -type d | sort | head -80 >&2 || true
    return 2
}

try_resolve_dataset_dir() {
    local root="$1"
    local preferred="$2"
    local fallback_pattern="$3"

    if [[ ! -d "${root}" ]]; then
        return 1
    fi
    if [[ -d "${root}/${preferred}" ]]; then
        printf '%s\n' "${root}/${preferred}"
        return 0
    fi

    local found
    found="$(find "${root}" -maxdepth 3 -type d -name "${preferred}" -print -quit 2>/dev/null || true)"
    if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}"
        return 0
    fi

    found="$(find "${root}" -maxdepth 3 -type d -iname "${fallback_pattern}" -print -quit 2>/dev/null || true)"
    if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}"
        return 0
    fi
    return 1
}

print_dataset_dir() {
    local label="$1"
    local path="$2"
    echo "${label}=${path}"
    if [[ ! -d "${path}" ]]; then
        echo "ERROR: ${label} is not a directory: ${path}" >&2
        return 2
    fi
}
