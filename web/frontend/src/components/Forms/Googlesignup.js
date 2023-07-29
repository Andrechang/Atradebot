import React from 'react'
import { GoogleLoginButton } from 'react-social-login-buttons';
import { LoginSocialGoogle } from 'reactjs-social-login';


export default function Googlesignup() {
  return (
    <div>
      <LoginSocialGoogle
        client_id={'356547024443-d5ti20figat8cu1rptigsjmjktqe06fm.apps.googleusercontent.com'}
        scope="openid profile email"
        discoveryDocs='claims_supported'
        access_type='offline'
        onResolve={({ provider, data }) => {
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
        onReject={(err) => {
          console.log(err);
        }}
      >
        <GoogleLoginButton />
      </LoginSocialGoogle>
    </div>
  )
}
