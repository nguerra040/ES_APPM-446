from polynomial import Polynomial

def test_polynomial_eq():
    a = Polynomial.from_string("-4 + x^2")
    b = Polynomial.from_string("x^2 - 4")
    assert a == b

def test_polynomial_eq2():
    a = Polynomial.from_string("-4 + x^2")
    b = Polynomial.from_string("-4 - x^2")
    assert a != b

def test_polynomial_addition():
    a = Polynomial.from_string("1 + 7*x^2")
    b = Polynomial.from_string("-3 - x + 2*x^2")
    c = Polynomial.from_string("-2 - x + 9*x^2")
    assert a + b == c

def test_polynomial_addition2():
    a = Polynomial.from_string("1 + 7*x^2")
    b = Polynomial.from_string("-3 - x^2 + 2*x^3")
    c = Polynomial.from_string("-2 + 6*x^2 + 2*x^3")
    assert a + b == c

def test_polynomial_addition2():
    a = Polynomial.from_string("-3 - x^2 + 2*x^3")
    b = Polynomial.from_string("1 + 7*x^2")
    c = Polynomial.from_string("-2 + 6*x^2 + 2*x^3")
    assert a + b == c

def test_polynomial_addition3():
    a = Polynomial.from_string("-3 -x -7*x^2")
    b = Polynomial.from_string("1 + 7*x^2")
    c = Polynomial.from_string("-2 - x")
    assert a + b == c

def test_polynomial_subtraction():
    a = Polynomial.from_string("-2 + 6*x^2 + 2*x^3")
    b = Polynomial.from_string("-3 - x^2 + 2*x^3")
    c = Polynomial.from_string("1 + 7*x^2")
    assert a - b == c

def test_polynomial_subtraction2():
    a = Polynomial.from_string("-2 - x")
    b = Polynomial.from_string("1 + 7*x^2")
    c = Polynomial.from_string("-3 -x -7*x^2")
    assert a - b == c

def test_polynomial_multiplication():
    a = Polynomial.from_string("4")
    b = Polynomial.from_string("2 - x + 3*x^2")
    c = Polynomial.from_string("8 - 4*x + 12*x^2")
    assert a * b == c

def test_polynomial_multiplication2():
    a = Polynomial.from_string("3 - 2*x^2 + x^3")
    b = Polynomial.from_string("2 - x + 3*x^2")
    c = Polynomial.from_string("6 + 4*x^3 - 3*x + 5*x^2 - 7*x^4 + 3*x^5")
    assert a * b == c

