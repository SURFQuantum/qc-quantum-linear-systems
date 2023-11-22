[![Coverage Status](https://coveralls.io/repos/github/SURFQuantum/qc-quantum-linear-systems/badge.svg?branch=main)](https://coveralls.io/github/SURFQuantum/qc-quantum-linear-systems?branch=main)

# quantum-linear-systems
Quantum algorithms to solve linear systems of equations.

## Additional Requirements
### Qiskit Linear Systems
In addition to the requirements in `requirements.txt` it is also necessary to manually install the
[linear solvers](https://github.com/anedumla/quantum_linear_solvers) package, as Qiskit has deprecated the HHL
algorithm implementations (see their [algorithms migration guide](https://qiskit.org/documentation/migration_guides/algorithms_migration.html)).

The linear solvers package can be installed by executing:
```
pip install git+https://github.com/anedumla/quantum_linear_solvers
```

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
