import React from 'react';
import SignupForm from './SignupForm';
import {GoogleLoginButton} from 'react-social-login-buttons';
import {LoginSocialGoogle} from 'reactjs-social-login';

const App = () => {

  return (

    <div className='container'>
      <SignupForm/>
      <h3>Sign up with Google</h3>
      <LoginSocialGoogle
      client_id={'356547024443-d5ti20figat8cu1rptigsjmjktqe06fm.apps.googleusercontent.com'}
      scope="openid profile email"
      discoveryDocs='claims_supported'
      access_type='offline'
      onResolve={({provider,data})=>{
        const userData = {
          username: data['name'],
          email: data['email'],
          type: "google"
        };
        fetch('/signup', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(userData),
        })
        console.log(userData);
      }}
      onReject={(err)=>{
        console.log(err);
      }}
      >
        <GoogleLoginButton/>
       </LoginSocialGoogle>
    </div>
  );
};

export default App;
