# quantum-linear-systems
Quantum algorithms to solve linear systems of equations.

## Classiq SDK
This project makes use of the [classiq](https://www.classiq.io/) SDK.
To use the `classiq` SKD the user must perform authentication.
For more info check out the [classiq docs](https://docs.classiq.io/latest/) page.

Authentication is as simple as:
```
import classiq

classiq.authenticate()
```
This will open a confirmation window in your default browser. 
After confirming the user code the classiq login screen appears.
Completing the login completes the authentication process.