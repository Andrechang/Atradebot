import React from 'react';
import './Home.css';
import { Link } from 'react-router-dom';


export default function Home() {
  
  return (
    <div className=''>   
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
    <div className="mission-box">
        <h1 className='mission-title'>OUR MISSION</h1>
        <h3 className='mission-header'>Lorem ipsum dolor sit, amet consectetur adipisicing elit. Vero architecto, aspernatur tempore obcaecati assumenda et! Adipisci repellat voluptatem aperiam deserunt.</h3>
        <div className="mission-container">
          <div className="mission1">
            <img src="/images/Layer_1.png" alt="" />
            <p className="mission-para">Lorem ipsum dolor sit, amet consectetur adipisicing elit. Soluta, commodi.</p>
          </div>
          <div className="mission1">
          <img src="/images/Layer_1.png" alt="" />
          <p className="mission-para">Lorem ipsum dolor sit, amet consectetur adipisicing elit. Soluta, commodi.</p>
          </div>
          <div className="mission1">
          <img src="/images/Layer_1.png" alt="" />
          <p className="mission-para">Lorem ipsum dolor sit, amet consectetur adipisicing elit. Soluta, commodi.</p>
          </div>
        </div>
      </div>
      <div className="row featurette row1">
      <div className="col-md-7 order-md-2">
        <h2 className="featurette-heading fw-normal lh-1 row1-title">AI INTEGRATION </h2> 
        <span className=" row1-header">WISEBUCK CHATBOT</span>
        <p className="lead row1-para">Another featurette? Of course. More placeholder content here to give you an idea of how this layout would work with some actual real-world content in place.</p>
        <Link to="/CHATBOT" className="row1-btn">
        EXPLORE CHATBOT
            </Link>
      </div>
      <div className="col-md-5 order-md-1">
        
      </div>
    </div>
    <div className="row featurette row2">
      <div className="col-md-5">
        <h2 className="featurette-heading fw-normal lh-1 row2-title">PEER TO PEER COMPETITON</h2>
        <span className="row2-header">INVESTMENT</span><br />
        <span className="row2-header">CHALLENGE</span>
        <p className="lead row2-para">Lorem ipsum dolor sit amet, consectetur adipisicing elit. Illo accusantium quas modi facere repudiandae labore iste itaque eligendi, placeat quis.</p>
        <Link to="/competing" className="row2-btn">
        EXPLORE CHALLENGE
            </Link>
      </div>
      <div className="col-md-7">
        
      </div>
    </div>

    <div className="row featurette row1">
      <div className="col-md-7 order-md-2">
        <h2 className="featurette-heading fw-normal lh-1 row1-title">LOREM ISPUM </h2> 
        <span className=" row1-header">REWARDS BASED INVESTING</span>
        <p className="lead row1-para">Another featurette? Of course. More placeholder content here to give you an idea of how this layout would work with some actual real-world content in place.</p>
        <Link to="/rewards" className="row1-btn">
        EXPLORE REWARDS
            </Link>
      </div>
      <div className="col-md-5 order-md-1">
        
      </div>
    </div>
    <div className="row featurette row2">
      <div className="col-md-5">
        <h2 className="featurette-heading fw-normal lh-1 row2-title">LOREM ISPUM</h2>
        <span className="row2-header">PAPER TRADING</span>
        <p className="lead row2-para">Lorem ipsum dolor sit amet, consectetur adipisicing elit. Illo accusantium quas modi facere repudiandae labore iste itaque eligendi, placeat quis.</p>
        <Link to="/paper" className="row2-btn">
        EXPLORE TRADING
            </Link>
      </div>
      <div className="col-md-7">
      </div>
    </div>
<div className="footer">
  <div className="left">
    <small>LoGo</small> <h3 className='left-h3'>WISEBUCK.AI</h3>
    <p className='left-p'>@WiseBuck.AI,2023</p>
    <img src="/images/line.png" alt="" className='line'/>
  </div>
  <div className="right">
      <h3 className='footer-feature'>Features</h3>
    <div className="right1">
      <Link to="/chatbot" className='footer-link1'>WiseBuck ChatBot</Link>
      <Link to="/rewards" className='footer-link2'>Rewards Base Investing</Link>
    </div>
    <div className="right2">
    <Link to="/investing" className='footer-link1'>Investment Challenges</Link>
    <Link to="/paper" className='footer-link2'>Paper Trading</Link>
    </div>
    <div className="right3">
      <h3 className="footer-follow">Follow Us</h3>
      <img src="/images/insta.png" alt="" className='footer-img'/>
      <img src="/images/twitter.png" alt="" className='footer-img'/>
      <img src="/images/facebook.png" alt="" className='footer-img'/>
    </div>
    <div className="right4">
      <h3 className='footer-join'>Join The Community</h3>
      <div className="discord">
      <img src="/images/discord.png" alt="" className='footer-img'/><h4 className='discord-learn'>LEARN TO GROW</h4>
      </div>
    </div>
    <div className="right5">
      <Link to="/contact" className='footer-contact'>CONTACT US</Link>
    </div>
  </div>
</div>
    </div>
  )
}