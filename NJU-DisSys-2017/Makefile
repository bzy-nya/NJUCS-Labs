export GO111MODULE=off
export GOPATH = $(PWD)

all: run_ass1 run_ass2 run_ass3

run_ass1:
	cd src/raft && go test -run Election

run_ass2:
	cd src/raft && go test -run FailNoAgree 
	cd src/raft && go test -run ConcurrentStarts
	cd src/raft && go test -run Rejoin
	cd src/raft && go test -run Backup

run_ass3:
	cd src/raft && go test -run Persist1
	cd src/raft && go test -run Persist2
	cd src/raft && go test -run Persist3

test_basic_agree:
	cd src/raft && go test -run BasicAgree

test:
	cd src/raft && go test -run 2C
