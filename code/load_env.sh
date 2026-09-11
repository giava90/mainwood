#!/bin/bash
# Load code/local.env WITHOUT overriding anything already in the environment.
#
# Sourcing local.env directly (`set -a; . ./local.env`) assigns unconditionally, so
#     MAINWOOD_INPUT_TEMPLATE=/other/path/ ./run_conversion.sh WOOD Surselva
# was silently ignored -- the file won. code/paths.py has always given the
# environment precedence, so the shell and Python disagreed about the same config.
#
# This keeps the two in step: environment first, then local.env, then the built-in
# default. Parsing matches paths.load_local_env -- `export ` prefix, ` #` trailing
# comments and one layer of surrounding quotes are all stripped.
load_local_env() {
    local file="${1:-local.env}"
    [ -f "$file" ] || return 0

    local raw line key val current
    while IFS= read -r raw || [ -n "$raw" ]; do
        line=${raw#"${raw%%[![:space:]]*}"}                 # left-trim
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in
            export\ *) line=${line#export }
                       line=${line#"${line%%[![:space:]]*}"} ;;
        esac
        case "$line" in *=*) ;; *) continue ;; esac

        key=${line%%=*}
        val=${line#*=}
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac

        val=${val%%" #"*}                                    # strip trailing comment
        val=${val#"${val%%[![:space:]]*}"}                   # trim
        val=${val%"${val##*[![:space:]]}"}
        case "$val" in                                       # strip one quote layer
            \"*\") val=${val#\"}; val=${val%\"} ;;
            \'*\') val=${val#\'}; val=${val%\'} ;;
        esac

        eval "current=\${$key-}"
        [ -n "$current" ] || export "$key=$val"
    done < "$file"
}
