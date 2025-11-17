import argparse
from training.trainer import ModelTrainer
from evaluation.comparator import ModelComparator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'framework'], required=True)
    parser.add_argument('--model', choices=['famm', 'ramm', 'am2f'])
    parser.add_argument('--dataset', choices=['tongji', 'ieee', 'industrial'])
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = ModelTrainer(args.model, args.dataset)
        trainer.train()
    elif args.mode == 'eval':
        comparator = ModelComparator(args.dataset)
        comparator.run_comparison()
    elif args.mode == 'framework':
        from models.am2f import AM2FFramework
        framework = AM2FFramework()
        framework.evaluate_framework()


if __name__ == '__main__':
    main()