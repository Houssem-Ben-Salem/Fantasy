.PHONY: install ingest refresh squad serve test backtest clean

install:
	pip install -r requirements.txt

ingest:
	python -m fplbrain.ingest --db data/fpl.db

refresh:
	python scripts/refresh.py --db data/fpl.db --horizon 6

squad:
	python -c "import sqlite3,warnings; warnings.filterwarnings('ignore'); \
	from fplbrain import models,optimize; c=sqlite3.connect('data/fpl.db'); \
	p,_=models.project_horizon(c,'2026-27',1,6,['2024-25','2025-26']); \
	r=optimize.initial_squad(p,list(range(1,7))); print(r.summary()); \
	print(r.squad[['web_name','position','team','now_cost','horizon_xp','in_xi']].to_string(index=False))"

serve:
	python -m fplbrain.mcp_server

test:
	python -m pytest tests/ -q

backtest:
	python -m fplbrain.backtest --db data/fpl.db --season 2025-26 --history 2024-25

clean:
	rm -rf data/cache data/fpl.db
