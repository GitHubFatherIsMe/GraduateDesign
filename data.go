package main

import (
	"database/sql"
	"fmt"

	_ "github.com/lib/pq"
)

var db *sql.DB

func init() {
	var err error
	db, err = sql.Open("postgres", "user=postgres dbname=postgres password=120123 sslmode=disable")
	if err != nil {
		panic(err)
	}
}

// Record respresent the struction of record
type Record struct {
	Uuid   string
	Mail   string
	Seq    string //file_name
	Pssm   string //file_name
	Result string //file_name
}

// Get return all the Record in table
func Get(uuid string) (record Record, err error) {
	fmt.Println(uuid)
	row := db.QueryRow("select * from records where uuid=$1", uuid)

	err = row.Scan(&record.Uuid, &record.Mail, &record.Seq, &record.Pssm, &record.Result)

	return
}

// Create insert record into the table
func Create(record Record) {
	db.QueryRow("insert into records values($1,$2,$3,$4,$5)", record.Uuid, record.Mail, record.Seq, record.Pssm, record.Result)
}
