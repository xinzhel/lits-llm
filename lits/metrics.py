from .base_llm import InferenceLogger
def get_inference_cost_metrics(root_dir, run_id='', exclude_idx=None, include_idx=None, exclude_roles_prefix=[], max_check=None, return_metrics=None, verbose=True):
    # Efficiency metrics
    metric_all = {'num_calls': None, 'num_batch_calls': None, 'num_flatten_calls': None, 'input_tokens': None, 'output_tokens': None, 'running_time': None,}
    inference_logger = InferenceLogger(run_id=run_id, root_dir=root_dir)
    inference_logger.set_include_idx(include_idx)
    inference_logger.set_exclude_idx(exclude_idx)
    inference_logger.set_max_check(max_check)
    inference_logger.set_return_metrics(return_metrics)
    metric_efficiency = inference_logger.get_metrics_by_role(exclude_roles_prefix=exclude_roles_prefix)
    # num_seconds = 0
    # for role_prefix in VALID_ROLES_PREFIX:
    #     if role_prefix in exclude_roles_prefix:
    #         continue
    #     kv_d = inference_logger.get_metrics_by_prefix(role_prefix)
    #     num_seconds += kv_d.get("running_time", 0)
    # num_hours = num_seconds / 3600
    for key in metric_efficiency:
        metric_all[key] = round(metric_efficiency[key], 2)
    # metric_all["total_hours"] = round(num_hours, 2)
    if verbose:
        # Efficiency metrics - Roles
        print("Efficiency metrics - Roles")
        inference_logger.print_metrics_for_all_role_prefixes()
        # Efficiency metrics - Search phases
        # print("\nEfficiency metrics - Search phases")
        # inference_logger.print_metrics_for_mcts_phases()

    if return_metrics:
        return inference_logger, {k: v for k, v in metric_all.items() if k in return_metrics}
    return inference_logger, metric_all