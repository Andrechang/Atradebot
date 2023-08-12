import React from 'react';
import './Home.css';
import { Link } from 'react-router-dom';

export default function Home() {
  
  return (
    <div className='body'>
      <div className="box">
        <h2 className='home-h2'>Welcome To</h2>
        <h1 className='home-h1'>WISEBUCK.AI</h1>
        <p className='home-h31'><i>The personal assistant for your</i></p>
        <p className='home-h32'><i>Investment Journey</i></p>
        <Link to="/signup" className="home-btn1">
        GET STARTED
            </Link>
        <Link to="/login" className="home-btn2">
              Login
            </Link>

      </div>

      
    </div>
  )
}
