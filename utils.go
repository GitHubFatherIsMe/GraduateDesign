package main

import (
	"fmt"
	"math/rand"
	"net/smtp"
	"strings"
	"time"
)

// GetRandomString generate random string used as uuid.
func GetRandomString(l int) string {
	str := "0123456789abcdefghijklmnopqrstuvwxyz"
	bytes := []byte(str)
	result := []byte{}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < l; i++ {
		result = append(result, bytes[r.Intn(len(bytes))])
	}
	return string(result)
}

// SendMail send email to notify user result processed.
func SendMail(receiver []string) {
	user := "1416006037@qq.com"
	link := "http://localhost:8000/home"
	auth := smtp.PlainAuth("", user, "bxqhbpzkadlrgcai", "smtp.qq.com")
	subject := "Prediction has finished!"
	nickname := "Tony"
	contentType := "Content-Type:text/plain;charset=UTF-8"
	body := fmt.Sprintf("Dear %v\nyour prediction submitted has finished done,please access the following site using unique Uuid generated to get the result!:\n%v", user, link)
	msg := []byte("To:" + strings.Join(receiver, ",") + "\r\nFrom:" + nickname +
		"<" + user + ">\r\nSubject:" + subject + "\r\n" + contentType + "\r\n\r\n" + body)
	err := smtp.SendMail("smtp.qq.com:25", auth, user, receiver, msg)
	if err != nil {
		fmt.Printf("send mail error: %v", err)
	}
}
