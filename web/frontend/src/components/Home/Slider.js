import React from 'react'
import './Slider.css'
import { Link } from 'react-router-dom'

export default function Slider() {
  return (
    <div>
    <div className='Slidercontain'>
      <div id="carouselExampleCaptions" className="carousel slide carousel-fade">
  <div className="carousel-indicators">
    <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="0" className="active" aria-current="true" aria-label="Slide 1"></button>
    <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="1" aria-label="Slide 2"></button>
    <button type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide-to="2" aria-label="Slide 3"></button>
  </div>
  <div className="carousel-inner">
    <div className="carousel-item active">
      <img src="/images/slider1.jpg" className="d-block w-50 sliderimg" alt="error"/>
      <div className="caption">
        <h2 className='sliderheader'>AI chatbot</h2>
        <p className='sliderpara'>Lorem ipsum, dolor sit amet consectetur adipisicing elit. Ducimus deleniti cumque id pariatur! Corporis, Lorem ipsum dolor sit amet consectetur adipisicing elit.inventore.</p>
        <Link to="/chatbot" className='sliderbutton'>Chatbot</Link>

      </div>
    </div>
    <div className="carousel-item">
      <img src="/images/slider2.jpg" className="d-block w-50 sliderimg" alt="error"/>
      <div className="caption">
        <h2 className='sliderheader'>Competing Aginst Freinds</h2>
        <p className='sliderpara'>Lorem ipsum, dolor sit amet consectetur adipisicing elit. Ducimus deleniti cumque id pariatur! Corporis, Lorem ipsum dolor sit amet consectetur adipisicing elit.inventore.</p>
        <Link to="/competing" className='sliderbutton'>Competing Aginst Friends</Link>  
            </div>
      
    </div>
    <div className="carousel-item">
      <img src="/images/slider3.jpg" className="d-block w-50 sliderimg" alt="error"/>
      <div className="caption">
        <h2 className='sliderheader'>Rewards</h2>
        <p className='sliderpara'>Lorem ipsum, dolor sit amet consectetur adipisicing elit. Ducimus deleniti cumque id pariatur! Corporis, Lorem ipsum dolor sit amet consectetur adipisicing elit.inventore.</p>
        <Link to="/rewards" className='sliderbutton'>Rewards</Link>      </div>
    </div>
  </div>
  <button className="carousel-control-prev" type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide="prev">
    <span className="carousel-control-prev-icon" aria-hidden="true"></span>
    <span className="visually-hidden">Previous</span>
  </button>
  <button className="carousel-control-next" type="button" data-bs-target="#carouselExampleCaptions" data-bs-slide="next">
    <span className="carousel-control-next-icon" aria-hidden="true"></span>
    <span className="visually-hidden">Next</span>
  </button>
</div>
    </div>
    </div>
  )
}
