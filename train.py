"""CLI entry point: python train.py [--epochs 50 --batch-size 2 ...]  (see patchdetr/config.py)."""
from patchdetr import Trainer, config_from_args


def main() -> None:
	cfg = config_from_args()
	trainer = Trainer(cfg).setup()
	trainer.fit()
	if cfg.train.run_test:
		print("[TEST]", {k: v for k, v in trainer.test().items() if k != "prior_diagnostics"})


if __name__ == "__main__":
	main()
