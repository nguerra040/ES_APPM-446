import pytest

from polynomial import Polynomial, RationalPolynomial

def test_rational_polynomial_eq():
    a = RationalPolynomial.from_string("(x+1)/(1-x)")
    b = RationalPolynomial.from_string("(1+x)/(-x+1)")
    assert a == b

def test_rational_polynomial_eq2():
    a = RationalPolynomial.from_string("(x+1)/(1)")
    b = RationalPolynomial.from_string("(x+1)/(x-1)")
    assert a != b

def test_rational_polynomial_reduce():
    a = RationalPolynomial.from_string("(2 + 3 * x + x^2)/(1 + 2 * x + x^2 + x^3 + x^4)")
    b = RationalPolynomial.from_string("(2 + x)/(1 + x + x^3)")
    assert a == b

def test_rational_polynomial_reduce2():
    a = RationalPolynomial.from_string("(-2 - 3 * x - x^2)/(1 + 2 * x + x^2 + x^3 + x^4)")
    b = RationalPolynomial.from_string("(2 + x)/(-1 - x - x^3)")
    assert a == b

def test_rational_polynomial_addition():
    a = RationalPolynomial.from_string("(1+2*x^2)/(-x)")
    b = RationalPolynomial.from_string("(2-x)/(x^2+5)")
    c = RationalPolynomial.from_string("(-5 + 2 * x - 12 * x^2 - 2 * x^4)/(5 * x + x^3)")
    assert a + b == c

def test_rational_polynomial_addition2():
    a = RationalPolynomial.from_string("(2 + 2 * x + 2 * x^2)/(2 + x)")
    b = RationalPolynomial.from_string("(2-x)/(- 2 + x + x^2)")
    c = RationalPolynomial.from_string("( - x + 2 * x^3 )/( - 2 + x + x^2 )")
    assert a + b == c

def test_rational_polynomial_subtraction():
    a = RationalPolynomial.from_string("(-5 + 2 * x - 12 * x^2 - 2 * x^4)/(5 * x + x^3)")
    b = RationalPolynomial.from_string("(2-x)/(x^2+5)")
    c = RationalPolynomial.from_string("(1+2*x^2)/(-x)")
    assert a - b == c

def test_rational_polynomial_subtraction2():
    a = RationalPolynomial.from_string("( - x + 2 * x^3 )/( - 2 + x + x^2 )")
    b = RationalPolynomial.from_string("(2-x)/(- 2 + x + x^2)")
    c = RationalPolynomial.from_string("(2 + 2 * x + 2 * x^2)/(2 + x)")
    assert a - b == c

def test_rational_polynomial_multiplication():
    a = RationalPolynomial.from_string("( - x + 2 * x^3 )/( - 2 + x + x^2 )")
    b = RationalPolynomial.from_string("(2-x)/(- 2 + x + x^2)")
    c = RationalPolynomial.from_string("( 2 * x - x^2 - 4 * x^3 + 2 * x^4 )/( - 4 + 4 * x + 3 * x^2 - 2 * x^3 - x^4 )")
    assert a * b == c

def test_rational_polynomial_multiplication2():
    a = RationalPolynomial.from_string("(- 2 + x + x^2)/(-5 + 2 * x - 12 * x^2 - 2 * x^4)")
    b = RationalPolynomial.from_string("(2*x^3+1)/(- 2 * x - x^2 + 4 * x^3 + 2 * x^4)")
    c = RationalPolynomial.from_string("( - 1 + x - 2 * x^3 + 2 * x^4 )/( 5 * x - 2 * x^2 + 2 * x^3 + 4 * x^4 - 22 * x^5 - 4 * x^7 )")
    assert a * b == c

def test_rational_polynomial_division():
    a = RationalPolynomial.from_string("(- 2 + x + x^2)/(-5 + 2 * x - 12 * x^2 - 2 * x^4)")
    b = RationalPolynomial.from_string("(- 2 * x - x^2 + 4 * x^3 + 2 * x^4)/(2*x^3+1)")
    c = RationalPolynomial.from_string("( - 1 + x - 2 * x^3 + 2 * x^4 )/( 5 * x - 2 * x^2 + 2 * x^3 + 4 * x^4 - 22 * x^5 - 4 * x^7 )")
    assert a / b == c

def test_rational_polynomial_division2():
    a = Polynomial.from_string("- 2 + x + x^2")
    b = Polynomial.from_string("-5 + 2 * x - 12 * x^2 - 2 * x^4")
    c = RationalPolynomial.from_string("(- 2 + x + x^2)/(-5 + 2 * x - 12 * x^2 - 2 * x^4)")
    assert a / b == c

