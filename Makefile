run:
	python story_visualizer/run.py

lint:
	isort story_visualizer; \
	black story_visualizer
