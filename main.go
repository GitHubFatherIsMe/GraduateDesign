package main

import (
	"fmt"
	"html/template"
	"io/ioutil"
	"math/rand"
	"net/http"
	"strconv"
)

// Index show the navigator page
func Index(w http.ResponseWriter, r *http.Request) {
	files := []string{"templates/layout.html", "templates/navbar.html", "templates/upload.index.html"}
	t, err := template.ParseFiles(files...)
	if err != nil {
		panic(err)
	}
	t.ExecuteTemplate(w, "layout", "")
}

// Process receive the form and call Predict,finally store the PSSM,result in database with unqiue uuid returned. User could use uuid getting their own result.
func Process(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "hello world!")
	mail := r.FormValue("mail")
	fmt.Fprintln(w, mail)
	file, _, err := r.FormFile("residue_seq")
	// var chan1, chan2 chan string
	var data []byte
	var pssm, pdb string
	if err == nil {
		data, err = ioutil.ReadAll(file)
		if err == nil {
			fmt.Fprintln(w, string(data))
			pssm, pdb = Predict(string(data))
		}
	}
	fmt.Fprintln(w, pssm)
	fmt.Fprintln(w, pdb)
	if pssm != "" && pdb != "" {
		var uuid string
		for i := 0; i < 8; i++ {
			uuid += strconv.Itoa(rand.Intn(10))
		}
		fmt.Println(uuid)
		newRecord := Record{
			Uuid:   uuid,
			Mail:   mail,
			Seq:    string(data),
			Pssm:   pssm,
			Result: pdb,
		}
		fmt.Println(newRecord)
		Create(newRecord)
		// receiver := []string{mail}
		// SendMail(receiver)
	}
}

// Result retrieve database and return record according to uuid
func Result(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	uuid := r.Form["uuid"][0]
	record, _ := Get(string(uuid))
	files := []string{"templates/layout.html", "templates/navbar.html", "templates/result.index.html"}
	t, err := template.ParseFiles(files...)
	if err != nil {
		panic(err)
	}
	record.Result = "static/models/molecules/pdb1a6m.ent"
	Pssm, _ := ioutil.ReadFile("static/models/PSSMs/1a6m.pssm")
	Seq, err := ioutil.ReadFile("static/models/FASTAs/1a6m.fasta")
	if err != nil {
		panic(err)
	}
	record.Pssm, record.Seq = string(Pssm), string(Seq)
	if record.Uuid == "" {
		t.ExecuteTemplate(w, "layout", false)
	} else {
		t.ExecuteTemplate(w, "layout", record)
	}
}

// Predict pass the seq to psi-blast for PSSM and send it to PyTorch and return PSSM and result(PDB).
func Predict(seq string) (PSSM, result string) {
	PSSM = "I am PSSM"
	result = "I am result"
	return
}

func main() {
	// the point in "./static" is so expersive and waste my half an hour!
	files := http.FileServer(http.Dir("./static"))
	http.Handle("/static/", http.StripPrefix("/static/", files))
	http.HandleFunc("/process", Process)
	http.HandleFunc("/home", Index)
	http.HandleFunc("/result", Result)
	http.ListenAndServe(":8000", nil)
}
