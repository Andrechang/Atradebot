
import React, { useState } from 'react';
import './Login.css';
import { Link, useNavigate } from 'react-router-dom';


const LoginPage = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
  });

  const navigate=useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    const userData = {
      username: formData.username,
      password: formData.password,
      type: "webForm",

    };

    fetch('/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',

      },
      body: JSON.stringify(userData),
    })
      .then((response) => response.json())
      .then((data) => {
        // Handle the response from the backend
        console.log(data.success);
        // Add Redirect to home page in the below conditional
        // if(data.success === true){
        // }
        if(data.success === true){
          navigate('/')
          alert('logged in succesfully');
   
        }
        else{
          navigate('/signup')
          alert('Could not login , please sign up again');

        }
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  return (
    <div className="login-container">
      <h2 className='login-header' >Login</h2>
      <form onSubmit={handleSubmit} className='form-controller'>
          
<div class="input-container">
    <i class="fa fa-user icon"></i>
    <input class="input-field" type="text" placeholder="Please enter your Username" name="username"
     value={formData.username}
     onChange={handleChange}
     required/>
  </div>
 
  
  <div class="input-container">
    <i class="fa fa-key icon"></i>
    <input class="input-field" type="password" placeholder="Please enter your Password" name="password"
     value={formData.password}
     onChange={handleChange}
     required/>
  </div>        
        <button type="submit" className='form-login-btn'>Login</button>
      </form>
      <h4 className='account'>Don't have an account? <Link to="/signup" className="account-btn">
              SIGN UP
            </Link></h4>
    </div>
  );
};

export default LoginPage;

