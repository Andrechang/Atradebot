import React from 'react';
import './Mcq.css';
import Demo from './Demo'

export default function Mcq() {
  const questions = [
    {
      question: "1-What is your age group?",
      options: ["Under 25", "25-34", "35-44", "45 or older"],
    },
    {
      question: "2- What is your investment learning goal?",
      options: ["To generate income in the short-term","To build wealth in the long-term", "To preserve my capital"],
    },
    {
      question: "3- How would you feel if your investments lost value in the short-term?",
      options: ["I would sell immediately", "I would hold onto my investments", "I would buy more"],
    },
    {
      question: "4- What is your current financial situation?",
      options: ["I'm struggling financially and living paycheck to paycheck", "I'm financially stable with a steady income and some savings", "I have a significant amount of savings and a high income","I have a lot of debt and financial obligations. "]
    }
    // Add more questions here...
  ];
  return (
    <div className='mcq-container'>
      <div className='mcq-box'>
      {questions.map((q, index) => (
        <Demo key={index} question={q.question} options={q.options} />
      ))} 
        <button type="button" className="btn btn-primary">Submit</button>
      </div>

    </div>
  )
      }



