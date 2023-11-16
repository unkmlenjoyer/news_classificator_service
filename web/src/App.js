import 'bootstrap/dist/css/bootstrap.min.css';
import React from 'react';
import './App.css';

import axios from 'axios';
import { useState } from "react";

function App() {

  const [data, setTitle] = useState('')
  const [score, setScore] = useState(new Map())
  const [news_table, setNewsTable] = useState([])
  const [error, setError] = useState("")

  const handleClick = async () => {
    axios.post(
      'http://0.0.0.0:5050/api/predict',
      { 'data': data }
    ).then(response => {
      if (response.status = 200) {
        setError("")
        return response
      } else {
        throw new Error("Sorry something went wrong")
      }
    })
      .then(text => {
        setScore(text['data']['scores'])
      })
      .catch(error => {
        setError(error.message)
      });
  }

  const handleClickHistory = async () => {
      axios.get(
        'http://0.0.0.0:5050/api/get_news_all',
    ).then(response => {
      if (response.status = 200) {
          return response
      } else {
        throw new Error("Can't catch news list")
      }
    })
      .then(data_table => {
        setNewsTable(data_table['data'])
      })
      .catch(error => {
        setError(error.message)
      });
  }


  return (
    <div className="App list-group-item  justify-content-center align-items-center mx-auto" style={{"backgroundColor": "white", "marginTop": "15px" }} >
      <div class="wrapper">
        <div class="box1">
            <nav class="navbar navbar-expand-lg" style={{"background-color": "#e3f2fd"}}>
            <a class="navbar-brand" href="#">Сервис классификации новостей</a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavAltMarkup" aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation">
              <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNavAltMarkup">
              <div class="navbar-nav">
                <a class="nav-item nav-link" href="https://github.com/unkmlenjoyer/news_classifier_service">Репозиторий проекта</a>
              </div>
            </div>
          </nav>
          <p></p>
          <div className="card-body">
          {error}
            <span className="card-text">
              <input className="mb-2 form-control titleIn mx-auto" style={{ "width": "1500px" }} onChange={event => setTitle(event.target.value)} placeholder='Пожалуйста, вставьте текст' />
              <button className="btn btn-outline-primary mx-2 mb-3" style={{ 'borderRadius': '50px', "font-weight": "bold" }} onClick={handleClick}>Предсказать категорию</button>
              <button className="btn btn-outline-primary mx-2 mb-3" style={{ 'borderRadius': '50px', "font-weight": "bold" }} onClick={handleClickHistory}>Показать историю запросов</button>
            </span>
          </div>
        </div>
        <div class="box2">
          <div>
            <table class="table table-hover table-bordered">
              <thead>
                <td>Категория</td>
                <td>Оценка</td>
              </thead>
              <tbody>
                {
                  Object.keys(score).map(key => {
                    return (
                        <tr>
                          <td>{key}</td>
                          <td>{(score[key] * 100).toFixed(3)}</td>
                        </tr>
                    );
                  })
                }
            </tbody>
            </table>
          </div>
        </div>
        <div class="box3">
          <div>
            <table class="table table-hover table-bordered">
                <thead>
                  <td>ID текста</td>
                  <td>Время запроса</td>
                  <td>Текстовая информация</td>
                  <td>Категория</td>
                </thead>
                <tbody>
                {news_table.sort((a, b) => a.insert_time < b.insert_time ? 1 : -1).slice(0, 10).map(item => {
                    return (
                      <tr>
                        <td>{item.text_id}</td>
                        <td>{ item.insert_time}</td>
                        <td>{ item.text.slice(0, 50) + '...' }</td>
                        <td>{ Object.entries(item.prediction).reduce((a, b) => a[1] > b[1] ? a : b)[0] }</td>
                      </tr>
                    );
                  })}
                </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

  );
}

export default App;
