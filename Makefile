# Submission for kaggle-stackoverflow challenge
# Marco Lui, October 2012
.PHONY: clean validate fullval
DATADIR=/lt/work/mlui/envs/kaggle-so/data
TRAIN=${DATADIR}/train_October_9_2012.csv
CHECK=${DATADIR}/train-tiny.csv
TEST=${DATADIR}/private_leaderboard.csv
VALIDATE=${DATADIR}/train-sample_October_9_2012_v2.csv

PARAMS=--loss_function logistic --oaa 5 --quiet ${ARGS}

submission.csv: test.pred
	python sigmoid_mc.py test.pred submission.csv

clean:
	rm -rf train.vw test.vw tiny.vw train.model test.pred submission.csv
	rm -rf validate.*

test.pred: model test.vw
	vw $(PARAMS) -i train.model -t -d test.vw -r test.pred

model: train.vw
	vw $(PARAMS) -d train.vw -f train.model

userdata.csv:
	python data2user.py $(TRAIN) userdata.csv

# This simply ensures that the model produced can be consumed by VW
check.vw: data2vw.py userdata.csv
	python data2vw.py $(CHECK) tiny.vw
	vw $(PARAMS) -d tiny.vw -f model.tiny

train.vw: data2vw.py check.vw userdata.csv
	python data2vw.py $(TRAIN) train.vw

test.vw: data2vw.py check.vw userdata.csv
	python data2vw.py $(TEST) test.vw

validate.vw:
	python data2vw.py $(VALIDATE) validate.vw

validate: validate.vw
	split -n r/10 validate.vw date.
	parallel "mv {} vali{}.fold" ::: date.*
	parallel "find . -name 'validate.*.fold' ! -name {} | xargs cat | vw $(PARAMS) -f {.}.model" ::: validate.*.fold
	parallel "vw $(PARAMS) -i {.}.model -t -d {} -r {.}.pred" ::: validate.*.fold
	parallel "python sigmoid_mc.py {} {.}.csv" ::: validate.*.pred
	parallel "python evaluate.py {.}.fold {}" ::: validate.*.csv | paste -sd+ | bc
	
fullval: train.vw
	split -n r/10 train.vw rain.
	parallel "mv {} t{}.fold" ::: rain.*
	parallel "find . -name 'train.*.fold' ! -name {} | xargs cat | vw $(PARAMS) -f {.}.model" ::: train.*.fold
	parallel "vw $(PARAMS) -i {.}.model -t -d {} -r {.}.pred" ::: train.*.fold
	parallel "python sigmoid_mc.py {} {.}.csv" ::: train.*.pred
	parallel "python evaluate.py {.}.fold {}" ::: train.*.csv | paste -sd+ | bc

