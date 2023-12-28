import wandb
import jax.numpy as jnp
import jax


def init_logger(args):
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            job_type="train_agents",
        )


def log_results(args, results):
    rets = results["metrics"]["returned_episode_returns"]

    # Hack to avoid logging 0 before first episodes are done
    # Required until https://github.com/RobertTLange/gymnax/issues/62 is resolved
    returned = results["metrics"]["returned_episode"]
    num_agents, num_train_steps = returned.shape
    """ REMOVED, MEAN TAKEN
    num_agents, num_train_steps, num_env_workers, num_rollout_steps = returned.shape
    first_episode_done = jnp.zeros((num_agents, num_env_workers), dtype=jnp.bool_)
    all_done_step = 0
    while not first_episode_done.all():
        step_episodes_done = jnp.any(returned[:, all_done_step], axis=-1)
        first_episode_done |= step_episodes_done
        all_done_step += 1
    """
    all_done_step = 0
    return_list = [
        rets[:, step].mean() for step in range(all_done_step, num_train_steps)
    ]

    if args.log:
        for step in range(rets.shape[1]):
            step_ret = None
            if step >= all_done_step:
                step_ret = return_list[step - all_done_step]
            log_dict = {
                "return": step_ret,
                "step": step,
                **{
                    k: v[:, step].mean()
                    for k, v in results["loss"].items()
                    if k not in {"grad_second_moment", "threshold_grad_second_moment"}
                },
                # Log dormancy for first agent only
                "dormancy": {
                    **{
                        k: v[0, step].mean()
                        for k, v in results["metrics"]["dormancy"].items()
                    },
                },
            }
            if args.log_gsm:
                grad_second_moment = {}
                for k, v in results["loss"]["grad_second_moment"]["params"].items():
                    grad_second_moment[k] = {
                        kk: wandb.Histogram(
                            np_histogram=(vv[0][0, step], vv[1][0, step])
                        )
                        for kk, vv in v.items()
                    }
                log_dict["grad_second_moment"] = grad_second_moment
                log_dict["threshold_grad_second_moment"] = jax.tree_map(
                    lambda x: x[:, step].mean(),
                    results["loss"]["threshold_grad_second_moment"],
                )
            wandb.log(log_dict)
    else:
        print("Step returns:", jnp.around(jnp.array(return_list), decimals=2))
