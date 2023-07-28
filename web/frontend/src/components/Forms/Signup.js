import React from 'react';
import GoogleSignUp from './GoogleSignUp';
import SignupForm from './SignupForm';


const App = () => {
  const handleGoogleSignInSuccess = (userData) => {
    // Handle successful Google Sign-In
    console.log('User Data:', userData);
  };

  const handleGoogleSignInFailure = () => {
    // Handle Google Sign-In failure
    console.log('Google Sign-In failed');
  };

  return (
    <div className='container'>
        <SignupForm/>
      <h3>Sign Up with Google</h3>
      <GoogleSignUp
        onGoogleSignInSuccess={handleGoogleSignInSuccess}
        onGoogleSignInFailure={handleGoogleSignInFailure}
      />
    
    </div>
  );
};

export default App;
