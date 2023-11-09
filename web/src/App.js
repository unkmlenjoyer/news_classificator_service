import 'bootstrap/dist/css/bootstrap.min.css';
import React from 'react';
import './App.css';

import axios from 'axios';
import { useState } from "react";

function App() {

  const [data, setTitle] = useState('')
  const [score, setScore] = useState(new Map())
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
        console.log(text)
        setScore(text['data']['scores'])
      })
      .catch(error => {
        setError(error.message)
      });
  }
  return (
    <div className="App list-group-item  justify-content-center align-items-center mx-auto" style={{ "width": "400px", "backgroundColor": "white", "marginTop": "15px" }} >
      <h6 className="card text-white bg-primary mb-3">Классификатор новостей</h6>
      <div className="card-body">
      {error}
        <span className="card-text">
          <input className="mb-2 form-control titleIn" onChange={event => setTitle(event.target.value)} placeholder='News headline' />
          <button className="btn btn-outline-primary mx-2 mb-3" style={{ 'borderRadius': '50px', "font-weight": "bold" }} onClick={handleClick}>Предсказать категорию</button>
        </span>
      </div>
      <div>
        <table>
          <thead>
            <td>Category</td>
            <td>Probas</td>
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
      <div>
      <table>
          <thead>
            <td>Category</td>
            <td>Probas</td>
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
  );
}

export default App;
