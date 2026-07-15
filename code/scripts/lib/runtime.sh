#!/usr/bin/env bash

find_code_dir() {
    local dir="$1"
    while [[ "${dir}" != "/" && -n "${dir}" ]]; do
        if [[ -f "${dir}/run.py" && -d "${dir}/nowcasting" ]]; then
            printf '%s\n' "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    echo "ERROR: could not locate code directory from ${1}" >&2
    return 2
}

enter_code_dir() {
    CODE_DIR="$(find_code_dir "$1")"
    export CODE_DIR
    cd "${CODE_DIR}"
}

run_python_module() {
    python -u -m "$@"
}
