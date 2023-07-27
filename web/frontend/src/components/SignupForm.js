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

//   const [passwordError, setPasswordError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });

    // Password validation
    // if (name === 'password') {
    //   const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
    //   if (!passwordRegex.test(value)) {
    //     setPasswordError(
    //       'Password must be at least 8 characters long and contain one capital letter and one special character.'
    //     );
    //   } else {
    //     setPasswordError('');
    //   }
    // }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // You can perform form submission logic here, like sending data to the server.
    console.log(formData);
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
          {/* {passwordError && <p className="error">{passwordError}</p>} */}
        </div>
        <button type="submit" className='btn btn-primary signupformbtn'>
          SignUp
        </button>
        <p>Already have an account? <Link to='/login'>Login</Link></p>
        <h3 className='or'>Or?</h3>
      </form>
    </div>
  );
};

export default SignUpForm;
