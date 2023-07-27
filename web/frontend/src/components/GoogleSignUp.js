import React from 'react';
import { GoogleLogin } from 'react-google-login';

const GoogleSignUp = ({ onGoogleSignInSuccess, onGoogleSignInFailure }) => {
  const clientId = '356547024443-d5ti20figat8cu1rptigsjmjktqe06fm.apps.googleusercontent.com';

  const responseGoogle = (response) => {
    // Handle the response from Google Sign-In
    if (response && response.profileObj) {
      onGoogleSignInSuccess(response.profileObj);
    } else {
      onGoogleSignInFailure();
    }
  };

  return (
    <GoogleLogin
      clientId={clientId}
      buttonText="Sign up with Google"
      onSuccess={responseGoogle}
      onFailure={responseGoogle}
      cookiePolicy={'single_host_origin'}
    />
  );
};

export default GoogleSignUp;
