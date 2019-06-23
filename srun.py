import subprocess
import argparse
import textwrap


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='srun')
    parser.add_argument('script', type=str)
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--partition', type=str, default='16gV100')
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=8)
    args = parser.parse_args()

    command = textwrap.dedent(f'''\
        srun \
        --mpi=pmi2 \
        --partition={args.partition} \
        --nodes={args.num_nodes} \
        --ntasks-per-node={args.num_gpus} \
        --ntasks={args.num_nodes * args.num_gpus} \
        --gres=gpu:{args.num_gpus} \
        python -u {args.script} \
        --config={args.config} \
        --checkpoint={args.checkpoint} \
        --training \
        --validation \
    ''')

    subprocess.call(command.split())
