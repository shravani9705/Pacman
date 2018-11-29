import React from 'react';
import ReactDOM from 'react-dom';
import {Login, LoginWithGoogle} from './login.js'
import 'bootstrap/dist/css/bootstrap.css';
import UserProfile from './userProfile.js';
import './style.css';
import firebase, {auth, provider} from './firebase.js';
import {LoginModals} from './loginModals.js';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      home: false,
      recipe: false,
      user: true,
      userState: 0,
      error: 0,
      google: 0
    }
    this.handleClick = this.handleClick.bind(this);
    this.recipeHandleClick = this.recipeHandleClick.bind(this);
    this.userHandleClick = this.userHandleClick.bind(this);
    this.loginWithGoogle = this.loginWithGoogle.bind(this);
    this.logoutWithGoogle = this.logoutWithGoogle.bind(this);
    this.logout = this.logout.bind(this);
    this.login = this.login.bind(this);
    this.signup = this.signup.bind(this);
    // this.componentDidMount = this.componentDidMount.bind(this)
  }
  componentDidMount() {
    auth.onAuthStateChanged((userState) => {
      if (userState) {
        this.setState({userState});
      }
    });
  }
  logoutWithGoogle() {
    this.setState({google: 0})
    auth.signOut().then(() => {
      this.setState({userState: 0});
    });
  }
  loginWithGoogle() {
    this.setState({google: 1});
    auth.signInWithPopup(provider).then((result) => {
      const userState = result.user;
      // console.log(userState)
      this.setState({userState});
      // console.log(this.state.userState)
    });
  }

  logout() {
    if (this.state.google == 1) {
      this.logoutWithGoogle();
    } else {
      firebase.auth().signOut().then(() => {
        this.setState({userState: 0});
      }).catch(function(error) {
        console.log(error);
      });
    }
  }

  signup(email) {
    this.setState({
      userState: {
        email: email
      }
    });
    // console.log('signup');
  }

  login(email) {
    this.setState({email: email})
    let dataRef = 'user-info/' + this.state.email
    console.log("The database referred to:")
    console.log(dataRef)
    let userRef = firebase.database().ref(dataRef);
    userRef.on("value", (snapshot) => {
      let items = snapshot.val();
      this.setState({
        userState: {
          displayName: items['name'],
          photoURL: '',
          email: email
        }
      })
      // console.log("this.state:")
      // console.log(this.state)
    }, function(errorObject) {
      console.log("The read failed: " + errorObject.code);
    });
    return 0
  }

  handleClick() {
    this.setState(state => ({home: true}));
    this.setState(state => ({recipe: false}));
    this.setState(state => ({user: false}));

  }

  recipeHandleClick() {
    this.setState(state => ({home: false}));
    this.setState(state => ({recipe: true}));
    this.setState(state => ({user: false}));
  }

  userHandleClick() {
    this.setState(state => ({home: false}));
    this.setState(state => ({recipe: false}));
    this.setState(state => ({user: true}));
  }

  render() {
    if (this.state.home) {
      return (< html > <div >
        <header align="center">
          <a href="#" onClick={this.handleClick
}>
            Home Page < /a> < a href = "#" onClick = {this.recipeHandleClick}
            > Recipe Page < /a >
            <a href="#" onClick={this.userHandleClick
}>
              User Page < /a> < /header >
              <HomePage >
                < /div>
                  < /html >); } else if (this.state.recipe) {
                      return (< div > <header align="center">
                        <a href="#" onClick={this.handleClick
}>
                          Home Page < /a> < a href = "#" onClick = {this.recipeHandleClick}
                          > Recipe Page < /a >
                          <a href="#" onClick={this.userHandleClick
}>
                            User Page < /a> < /header >
                            <RecipePage >
                              < /div>
                                );
    }
                    else if (this.state.user) {
                      // console.log('in App')
                      // console.log(this.state.userState)
                      return (< div > <header align="center">
                        <a href="#" onClick={this.handleClick
}>
                          Home Page < /a >
                          <a href="#" onClick={this.recipeHandleClick
}>
                            Recipe Page < /a >
                            <a href="#" onClick={this.userHandleClick
}>
                              User Page < /a > < /header >
                              <LoginWithGoogle userState={this.state.userState
} login={this.loginWithGoogle
} logout={this.logoutWithGoogle
} google={this.state.google
}>
                                <LoginModals buttonLabel='Login' signup={this.signup
} login={this.login
} logout={this.logout
} userState={this.state.userState
} google={this.state.google
}>
                                  <UserPage userState={this.state.userState
}/>
                                  < / div="div">
                                    );
    }
                    } } class HomePage extends React.Component {
                      constructor(props) {
                        super(props);
                      }

                      render() {
                        return (< div > Home < /div >);
    }
  }
                    class RecipePage extends React.Component {
                      constructor(props) {
                        super(props);
                      }

                      render() {
                        return (< div > Recipe < /div>
    );
  }
}

                    class UserPage extends React.Component {
                      constructor(props) {
                        super(props);
                        this.state = {
                          userId: -1
                        }
                        // console.log('in UserPage')
                        // console.log(this.state)
                        // console.log(this.props)
                      }
                      render() {
                        if (this.props.userState) {
                          return (< div > User Page < UserProfile userState = {
                            this.props.userState
                          } / > < /
          div >
        );
      } else {
        return ( <
          div >
          User Page <
          /div >);

                        }
                        // alert(this.prop.userState)

                      }
                    }

                    ReactDOM.render(< App / >, document.getElementById("root"));
