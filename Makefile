.PHONY: setup doctor train-all train-parallel eval-all assemble validate ready \
       clean dvc-push dvc-pull plot-all help

PYTHON ?= python3
TECHNIQUES := ppo a2c dqn ddqn reinforce a3c tabular_qlearning tabular_sarsa td_lambda nn_qlearning nn_sarsa

# ---- Help (default) ----

help:
	@echo ""
	@echo "  RL Inventory Control Pipeline"
	@echo "  ============================="
	@echo ""
	@echo "  Setup:"
	@echo "    make setup              Install dependencies, create directories"
	@echo "    make doctor             Check environment for issues (run on new machines)"
	@echo "    make migrate            Move existing flat models into pipeline structure"
	@echo ""
	@echo "  Training:"
	@echo "    make train-ppo          Train one technique (skip if done)"
	@echo "    make train-all          Train all sequentially"
	@echo "    make train-parallel     Train all in background"
	@echo "    make train-ppo FORCE=1  Retrain from scratch"
	@echo "    make train-ppo TS=2000000  Override timesteps"
	@echo ""
	@echo "  Evaluation:"
	@echo "    make eval-ppo           Evaluate one, save CSV"
	@echo "    make eval-all           Evaluate all + comparison table"
	@echo "    make eval-ppo FORCE=1   Re-evaluate"
	@echo ""
	@echo "  Submission:"
	@echo "    make assemble           Build upload-ready zips"
	@echo "    make validate           Run policy validation tests"
	@echo "    make ready              assemble + validate (pre-upload)"
	@echo ""
	@echo "  Monitoring:"
	@echo "    make plot-ppo           Live training plot"
	@echo "    make plot-all           Live plot all techniques"
	@echo "    make plot-save          Save training plots as PNG"
	@echo "    make check-ppo          Check if technique has converged"
	@echo "    make check-all          Check convergence for all techniques"
	@echo ""
	@echo "  Sync:"
	@echo "    make dvc-push           Push models to Google Drive"
	@echo "    make dvc-pull           Pull models from Google Drive"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean-ppo          Remove one technique's artifacts"
	@echo "    make clean              Remove all training artifacts"
	@echo ""

# ---- Setup ----

setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	@mkdir -p models plots results submissions
	@for t in $(TECHNIQUES); do mkdir -p models/$$t/checkpoints submissions/$$t; done
	@echo "Setup complete. Running environment check..."
	@$(MAKE) doctor

migrate:
	$(PYTHON) migrate_existing.py --steps 5000000

doctor:
	$(PYTHON) check_env.py

# ---- Training (pattern rule) ----

FORCE_FLAG := $(if $(FORCE),--force,)
TS_FLAG := $(if $(TS),--timesteps $(TS),)

train-%:
	$(PYTHON) train.py $* $(FORCE_FLAG) $(TS_FLAG)

train-all:
	$(PYTHON) train.py all $(FORCE_FLAG) $(TS_FLAG)

train-parallel:
	@for t in $(TECHNIQUES); do \
		echo "Starting $$t in background..."; \
		$(PYTHON) train.py $$t $(FORCE_FLAG) $(TS_FLAG) &  \
	done; \
	echo "All training jobs launched. Use 'make plot-all' to monitor."; \
	wait; \
	echo "All training complete."

# ---- Evaluation (pattern rule) ----

EVAL_FORCE := $(if $(FORCE),--force,)

eval-%:
	$(PYTHON) evaluate.py $* $(EVAL_FORCE)

eval-all:
	$(PYTHON) evaluate.py all $(EVAL_FORCE)

# ---- Submission ----

assemble:
	$(PYTHON) assemble.py

validate:
	$(PYTHON) validate.py

ready: assemble validate
	@echo ""
	@echo "All submissions ready. Upload *_upload.zip files from submissions/."

# ---- Plotting ----

plot-%:
	$(PYTHON) plot_live.py $*

plot-all:
	$(PYTHON) plot_live.py all

plot-save:
	$(PYTHON) plot_live.py all --save

# ---- Convergence check ----

check-%:
	$(PYTHON) check_convergence.py $*

check-all:
	$(PYTHON) check_convergence.py all

# ---- Leaderboard logging ----

log:
	@test -n "$(COST)" || (echo "Usage: make log COST=268856 NOTES='initial'" && exit 1)
	$(PYTHON) leaderboard_log.py --cost $(COST) --change "$(CHANGE)" --notes "$(NOTES)"

log-show:
	$(PYTHON) leaderboard_log.py --show

# ---- DVC / Git sync ----

dvc-push:
	dvc add models/
	git add models.dvc .gitignore
	@if [ -n "$$(git status --porcelain)" ]; then \
		git commit -m "Update models"; \
	fi
	git push
	dvc push

dvc-pull:
	git pull
	dvc pull

# ---- Cleanup ----

clean-%:
	rm -rf models/$*/
	rm -f results/$*_eval.csv
	mkdir -p models/$*/checkpoints
	@echo "Cleaned $*"

clean:
	rm -rf models/*/checkpoints/* models/*/final_model* models/*/best_model* \
	       models/*/train_log.csv models/*/training_metadata.json \
	       models/*/hyperparams_used.yaml
	rm -rf results/* plots/*
	@echo "Cleaned all artifacts. Models directory structure preserved."
