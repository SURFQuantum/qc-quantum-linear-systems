# quantum-linear-systems
Quantum algorithms to solve linear systems of equations.

## Installation
Dependencies are managed with [poetry](https://python-poetry.org/).
If poetry is not already installed on your system/evironment, follow the [installation instructions](https://python-poetry.org/docs/#installation).

Then clone the project
```
git clone https://github.com/SURFQuantum/qc-quantum-linear-systems.git
```
Switch to the project directory with ```cd qc-quantum-linear-systems``` and then simply:
```
poetry install
```
to install all dependencies.
## Additional Requirements
### Classiq SDK
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
