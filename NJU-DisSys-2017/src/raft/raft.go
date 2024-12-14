package raft

//
// this is an outline of the API that raft must expose to
// the service (or tester). see comments below for
// each of these functions for more details.
//
// rf = Make(...)
//   create a new Raft server.
// rf.Start(command interface{}) (index, term, isleader)
//   start agreement on a new log entry
// rf.GetState() (term, isLeader)
//   ask a Raft for its current term, and whether it thinks it is leader
// ApplyMsg
//   each time a new entry is committed to the log, each Raft peer
//   should send an ApplyMsg to the service (or tester)
//   in the same server.
//

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"labrpc"
	"math/rand"
	"slices"
	"sync"
	"time"
)

// import "bytes"
// import "encoding/gob"

const (
	Follower  = "Follower"
	Candidate = "Candidate"
	Leader    = "Leader"
)

// as each Raft peer becomes aware that successive log entries are
// committed, the peer should send an ApplyMsg to the service (or
// tester) on the same server, via the applyCh passed to Make().
type ApplyMsg struct {
	Index       int
	Command     interface{}
	UseSnapshot bool   // ignore for lab2; only used in lab3
	Snapshot    []byte // ignore for lab2; only used in lab3
}

type Entry struct {
	Index   int
	Term    int
	Command interface{}
}

// A Go object implementing a single Raft peer.
type Raft struct {
	mu        sync.Mutex
	peers     []*labrpc.ClientEnd
	persister *Persister
	me        int // index into peers[]

	// Your data here.
	// Look at the paper's Figure 2 for a description of what
	// state a Raft server must maintain.
	node_state   string // "leader", "follower", "candidate"
	current_term int
	voted_for    int

	applyCh chan ApplyMsg

	logs         []Entry
	commit_index int
	last_applied int
	next_index   []int
	match_index  []int

	timer *time.Timer
}

func (rf *Raft) String() string {
	return fmt.Sprintf("Node[%d, Term=%d]", rf.me, rf.current_term)
}

func HearbeatTimeout() time.Duration {
	return time.Duration(50) * time.Millisecond
}

func ElectionTimeout() time.Duration {
	return time.Duration(150+rand.Intn(150)) * time.Millisecond
}

func (rf *Raft) changeToFollower(term int) {
	rf.node_state = Follower
	rf.voted_for = -1

	if term != -1 {
		rf.current_term = term
	}

	rf.timer.Reset(ElectionTimeout())
}

func (rf *Raft) changeToCandidate() {
	rf.node_state = Candidate
	rf.current_term = rf.current_term + 1
	rf.timer.Reset(ElectionTimeout())
}

func (rf *Raft) changeToLeader() {
	rf.node_state = Leader
	rf.timer.Reset(HearbeatTimeout())
	rf.broadcastHeartbeat()
	for peer := range rf.peers {
		rf.match_index[peer] = 0
		rf.next_index[peer] = len(rf.logs)
	}
}

func (rf *Raft) OnTimeout() {
	for {
		select {
		case <-rf.timer.C:
			rf.mu.Lock()

			if rf.node_state == Leader {
				rf.broadcastHeartbeat()
				rf.timer.Reset(HearbeatTimeout())
			} else {
				rf.changeToCandidate()
				rf.raiseElection()
				rf.timer.Reset(ElectionTimeout())
			}

			rf.mu.Unlock()
		}
	}
}

func (rf *Raft) apply() {
	for {
		time.Sleep(10 * time.Millisecond)
		rf.mu.Lock()

		if rf.last_applied < rf.commit_index {
			DPrintf("%v appling entry[%v-%v]", rf, rf.last_applied+1, rf.commit_index)

			for i := rf.last_applied + 1; i <= rf.commit_index; i++ {
				rf.applyCh <- ApplyMsg{Index: i, Command: rf.logs[i].Command}
				rf.last_applied = i
			}
		}

		rf.mu.Unlock()
	}
}

func (rf *Raft) leaderCommit() {
	for {
		time.Sleep(10 * time.Millisecond)
		rf.mu.Lock()

		if rf.node_state == Leader {
			rf.next_index[rf.me] = len(rf.logs)
			rf.match_index[rf.me] = len(rf.logs) - 1

			var match_index []int = make([]int, len(rf.peers))

			copy(match_index, rf.match_index)
			slices.Sort(match_index)

			if rf.commit_index != match_index[len(rf.peers)/2] {
				DPrintf("%v commit to entry %v", rf, rf.commit_index)

				rf.commit_index = match_index[len(rf.peers)/2]
			}
		}

		rf.mu.Unlock()
	}
}

func (rf *Raft) broadcastHeartbeat() {
	DPrintf("%v broadcast heartbeat", rf)

	for peer := range rf.peers {
		if peer == rf.me {
			continue
		}
		go func(peer int) {
			rf.mu.Lock()
			var reply AppendEntriesReply
			var entries = make([]Entry, 0)
			var next = rf.next_index[peer]

			if rf.match_index[peer] == rf.next_index[peer]-1 {
				next = min(next+10, len(rf.logs))
				entries = rf.logs[rf.next_index[peer]:next]
			}
			rf.mu.Unlock()

			if !rf.sendAppendEntries(
				peer,
				AppendEntriesArgs{
					Term:         rf.current_term,
					LeaderID:     rf.me,
					LeaderCommit: rf.commit_index,
					PrevLogIndex: rf.logs[rf.next_index[peer]-1].Index,
					PrevLogTerm:  rf.logs[rf.next_index[peer]-1].Term,
					Entries:      entries,
				},
				&reply,
			) {
				return
			}

			rf.mu.Lock()

			if reply.Term > rf.current_term {
				DPrintf(
					"%v find a new leader Node[%v, Term = %v]",
					rf, peer, reply.Term,
				)
				rf.changeToFollower(reply.Term)
			} else if !reply.Success {
				rf.next_index[peer] = reply.ConflictIndex
			} else {
				rf.next_index[peer] = next
				rf.match_index[peer] = rf.next_index[peer] - 1
			}

			rf.mu.Unlock()
		}(peer)
	}
}

func (rf *Raft) raiseElection() {
	DPrintf("%v raise an election", rf)
	grantedVotes := 1
	rf.voted_for = rf.me

	for peer := range rf.peers {
		if peer == rf.me {
			continue
		}
		go func(peer int) {
			var reply RequestVoteReply
			if !rf.sendRequestVote(
				peer,
				RequestVoteArgs{
					CandidateID:  rf.me,
					Term:         rf.current_term,
					LastLogTerm:  rf.LastLogTerm(),
					LastLogIndex: rf.LastLogIndex(),
				},
				&reply,
			) {
				return
			}

			DPrintf("%v receive vote reply [%v, %v]", rf, reply.Term, reply.VoteGranted)

			rf.mu.Lock()
			defer rf.mu.Unlock()

			if reply.Term > rf.current_term {
				rf.changeToFollower(reply.Term)
			}

			if rf.node_state == Candidate && reply.VoteGranted {
				grantedVotes += 1

				if grantedVotes > len(rf.peers)/2 {
					DPrintf("%v win election", rf)

					defer rf.changeToLeader()
				}
			}
		}(peer)
	}
}

// return currentTerm and whether this server
// believes it is the leader.
func (rf *Raft) GetState() (int, bool) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	var term int = rf.current_term
	var isleader bool = rf.node_state == Leader

	return term, isleader
}

func (rf *Raft) LastLogTerm() int {
	return rf.logs[len(rf.logs)-1].Term
}

func (rf *Raft) LastLogIndex() int {
	return rf.logs[len(rf.logs)-1].Index
}

// save Raft's persistent state to stable storage,
// where it can later be retrieved after a crash and restart.
// see paper's Figure 2 for a description of what should be persistent.
func (rf *Raft) persist() {
	w := new(bytes.Buffer)
	e := gob.NewEncoder(w)
	e.Encode(rf.current_term)
	e.Encode(rf.voted_for)
	e.Encode(rf.logs)
	data := w.Bytes()
	rf.persister.SaveRaftState(data)
}

// restore previously persisted state.
func (rf *Raft) readPersist(data []byte) {
	r := bytes.NewBuffer(data)
	d := gob.NewDecoder(r)
	d.Decode(&rf.current_term)
	d.Decode(&rf.voted_for)
	d.Decode(&rf.logs)
}

// example RequestVote RPC arguments structure.
type RequestVoteArgs struct {
	Term        int
	CandidateID int

	LastLogIndex int
	LastLogTerm  int
}

// example RequestVote RPC reply structurentriese.
type RequestVoteReply struct {
	Term        int
	VoteGranted bool
}

// example RequestVote RPC handler.
func (rf *Raft) RequestVote(args RequestVoteArgs, reply *RequestVoteReply) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	DPrintf("%v receive a vote request from Node[%v, Term = %v]", rf, args.CandidateID, args.Term)

	if args.Term < rf.current_term || (args.Term == rf.current_term && rf.voted_for != -1) {
		reply.Term = rf.current_term
		reply.VoteGranted = false
	} else {
		rf.changeToFollower(args.Term)
		reply.Term = rf.current_term
		reply.VoteGranted = false

		if args.LastLogTerm > rf.LastLogTerm() || (args.LastLogTerm == rf.LastLogTerm() && args.LastLogIndex >= rf.LastLogIndex()) {
			reply.VoteGranted = true
		}
	}
}

// example code to send a RequestVote RPC to a server.
// server is the index of the target server in rf.peers[].
// expects RPC arguments in args.
// fills in *reply with RPC reply, so caller should
// pass &reply.
// the types of the args and reply passed to Call() must be
// the same as the types of the arguments declared in the
// handler function (including whether they are pointers).
//
// returns true if labrpc says the RPC was delivered.
//
// if you're having trouble getting RPC to work, check that you've
// capitalized all field names in structs passed over RPC, and
// that the caller passes the address of the reply struct with &, not
// the struct itself.
func (rf *Raft) sendRequestVote(server int, args RequestVoteArgs, reply *RequestVoteReply) bool {
	ok := rf.peers[server].Call("Raft.RequestVote", args, reply)
	return ok
}

type AppendEntriesArgs struct {
	Term int

	LeaderID     int
	PrevLogIndex int
	PrevLogTerm  int
	Entries      []Entry
	LeaderCommit int
}

type AppendEntriesReply struct {
	Term    int
	Success bool

	ConflictIndex int
}

func (rf *Raft) AppendEntries(args AppendEntriesArgs, reply *AppendEntriesReply) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	DPrintf("%v receive APR[%v, %v, %v] from Leader[%v, Term=%v]",
		rf,
		args.PrevLogIndex, args.PrevLogTerm, len(args.Entries),
		args.LeaderID, args.Term,
	)

	if args.Term < rf.current_term {
		reply.Term = rf.current_term
		reply.Success = false

		return
	}

	rf.changeToFollower(args.Term)
	reply.Term = rf.current_term

	if args.PrevLogIndex > rf.LastLogIndex() {
		reply.Success = false
		reply.ConflictIndex = len(rf.logs)

		return
	}

	if rf.logs[args.PrevLogIndex].Term != args.PrevLogTerm {
		reply.Success = false

		for i := args.PrevLogIndex; i >= 0; i-- {
			if rf.logs[args.PrevLogIndex] != rf.logs[i] {
				reply.ConflictIndex = i + 1
				break
			}
		}

		return
	}

	rf.logs = rf.logs[:args.PrevLogIndex+1]
	rf.logs = append(rf.logs, args.Entries...)

	rf.commit_index = min(rf.LastLogIndex(), args.LeaderCommit)

	rf.persist()

	reply.Term = rf.current_term
	reply.Success = true
}

func (rf *Raft) sendAppendEntries(server int, args AppendEntriesArgs, reply *AppendEntriesReply) bool {
	ok := rf.peers[server].Call("Raft.AppendEntries", args, reply)
	return ok
}

// the service using Raft (e.g. a k/v server) wants to start
// agreement on the next command to be appended to Raft's log. if this
// server isn't the leader, returns false. otherwise start the
// agreement and return immediately. there is no guarantee that this
// command will ever be committed to the Raft log, since the leader
// may fail or lose an election.
//
// the first return value is the index that the command will appear at
// if it's ever committed. the second return value is the current
// term. the third return value is true if this server believes it is
// the leader.
func (rf *Raft) Start(command interface{}) (int, int, bool) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	if rf.node_state != Leader {
		return -1, -1, false
	}

	DPrintf("%v receives a new command[%v]", rf, command)

	rf.logs = append(rf.logs, Entry{Index: rf.LastLogIndex() + 1, Term: rf.current_term, Command: command})

	rf.persist()

	defer rf.broadcastHeartbeat()

	return rf.LastLogIndex(), rf.current_term, true
}

// the tester calls Kill() when a Raft instance won't
// be needed again. you are not required to do anything
// in Kill(), but it might be convenient to (for example)
// turn off debug output from this instance.
func (rf *Raft) Kill() {
	// Your code here, if desired.
}

// the service or tester wants to create a Raft server. the ports
// of all the Raft servers (including this one) are in peers[]. this
// server's port is peers[me]. all the servers' peers[] arrays
// have the same order. persister is a place for this server to
// save its persistent state, and also initially holds the most
// recent saved state, if any. applyCh is a channel on which the
// tester or service expects Raft to send ApplyMsg messages.
// Make() must return quickly, so it should start goroutines
// for any long-running work.

func Make(peers []*labrpc.ClientEnd, me int,
	persister *Persister, applyCh chan ApplyMsg) *Raft {
	rf := &Raft{}
	rf.peers = peers
	rf.persister = persister
	rf.me = me
	rf.applyCh = applyCh

	// Your initialization code here.
	rf.current_term = 0
	rf.voted_for = -1
	rf.node_state = Follower

	rf.logs = make([]Entry, 1)
	rf.commit_index = 0
	rf.last_applied = 0
	rf.next_index = make([]int, len(rf.peers))
	rf.match_index = make([]int, len(rf.peers))

	rf.timer = time.NewTimer(ElectionTimeout())

	DPrintf("%v wake up", rf)

	// initialize from state persisted before a crash
	if len(rf.persister.raftstate) > 0 {
		rf.readPersist(persister.ReadRaftState())
	}
	rf.persist()

	go rf.OnTimeout()
	go rf.apply()
	go rf.leaderCommit()

	return rf
}
