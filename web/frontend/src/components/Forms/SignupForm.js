import React, { useState } from 'react';
import './SignupForm.css';
import { Link } from 'react-router-dom';

const SignUpForm = () => {
  const [formData, setFormData] = useState({
    name: '',
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Convert the form data to a JSON object
    const userData = {
      username: formData.username,
      email: formData.email,
      password: formData.password,
    };

    // Send the JSON object to the backend
    fetch('/api/signup', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(userData),
    })
      .then((response) => response.json())
      .then((data) => {
        // Handle the response from the backend
        console.log(data);
      })
      .catch((error) => {
        console.error('Error:', error);
      });
  };

  return (
    <div className='container'>
      <h2>Sign Up</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="username">Username:</label> <br />
          <input
            type="text"
            id="username"
            name="username"
            value={formData.username}
            onChange={handleChange}
            required
            className='input'
          />
        </div>
        <div>
          <label htmlFor="email">Email:</label> <br />
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            required
            className='input'
          />
        </div>
        <div>
          <label htmlFor="password">Password:</label> <br />
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
            className='input'
          />
        </div>
        <button type="submit" className='btn btn-primary signupformbtn'>
          SignUp
        </button>
        <p>Already have an account? <Link to='/login'>Log in</Link></p>
        <h3 className='or'>Or?</h3>
      </form>
    </div>
  );
};

export default SignUpForm;
