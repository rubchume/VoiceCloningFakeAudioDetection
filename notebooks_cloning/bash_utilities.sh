parse_arguments() {
    local argument_names=()
    local getopt_string=""
    local -A argument_variable_mapping
    
    for ((i = 1; i <= $#; i += 2)); do
        local j=$((1 + i))
        argument_name="${!i}"
        variable_name="${!j}"
        
        if [ "$argument_name" == "--" ] || [ "$variable_name" == "--" ]; then
            break
        fi
        
        getopt_string+="${argument_name}:,"
        argument_names+=("${argument_name}")
        argument_variable_mapping["$argument_name"]=$variable_name
    done
  
    getopt_string=${getopt_string%,}
    
    ARGS=$(getopt --long "$getopt_string" -- "" "${@: -$((${#argument_names[@]} * 2))}")
    eval set -- "$ARGS"
      
    local -A argument_value_mapping
    while true; do
        for arg_name in "${argument_names[@]}"; do
            if [ "$1" == "--${arg_name}" ]; then
              argument_value_mapping["$arg_name"]="$2"
              shift 2
              continue 2
            fi
        done
        if [ "$1" == "--" ]; then
            break
        else
            echo "Unexpected option: $1"
            exit 1
        fi
    done
  
    for arg_name in "${argument_names[@]}"; do
        if [ -n "${argument_value_mapping[$arg_name]}" ]; then
            variable_name=${argument_variable_mapping[$arg_name]}
            argument_value=${argument_value_mapping[$arg_name]}
            local -n outer_variable=$variable_name
            outer_variable=$argument_value
        fi
    done
}
