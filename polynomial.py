
import numpy as np
from sympy import cofactors
from fraction import FractionBase

class Polynomial:

    def __init__(self, order, coefficients, variable="x"):
        self.order = order
        self.coefficients = coefficients
        self.variable = variable

    def __repr__(self):
        # if zero
        if max(abs(self.coefficients)) == 0:
            return "0"
        string = ""
        for deg, coeff in enumerate(self.coefficients):
            if coeff != 0:
                if coeff > 0:
                    term = str(coeff)
                elif coeff < 0:
                    term = "- " + str(-coeff)
                if deg > 0:
                    if term == "1":
                        term = self.variable
                    elif term == "- 1":
                        term = "- " + self.variable
                    else:
                        term = term + " * " + self.variable
                if deg > 1:
                    term = term + "^" + str(deg)

                if string == "":
                    string = term
                else:
                    if coeff < 0:
                        string = string + " " + term
                    else:
                        string = string + " + " + term
        return string

    def __add__(self, other):
        if self.variable != other.variable:
            raise ValueError("Variables must be the same")

        # maximum possible order is the max of the two orders
        order = max(self.order, other.order)

        coefficients = np.zeros(order+1, dtype=int)

        # sum the coefficients
        # careful if polynomials have different degrees
        for deg in range(order+1):

            if deg <= self.order:
                coeff_self = self.coefficients[deg]
            else:
                coeff_self = 0
            if deg <= other.order:
                coeff_other = other.coefficients[deg]
            else:
                coeff_other = 0

            coefficients[deg] = coeff_self + coeff_other

        if max(abs(coefficients)) == 0:
            return self.from_string("0", self.variable)

        # check to see there was cancellation in the leading coefficients
        while coefficients[order] == 0:
            order -= 1
        coefficients = coefficients[:order+1]

        return Polynomial(order, coefficients, self.variable)

    def __neg__(self):
        coefficients = -1*self.coefficients
        return Polynomial(self.order, coefficients, self.variable)

    def __sub__(self, other):
        return self.__add__(-other)

    def __eq__(self, other):
        if self.variable == other.variable:
            if self.order == other.order:
                if (self.coefficients == other.coefficients).all():
                    return True
        return False

    def __mul__(self, other):
        if self.variable != other.variable:
            raise ValueError("Variables must be the same")

        order = self.order + other.order

        coefficients = np.zeros(order+1, dtype=int)

        # multiply each monomial in self separately
        for deg, coeff in enumerate(self.coefficients):
            coefficients[deg:deg+other.order+1] += coeff*other.coefficients

        return Polynomial(order, coefficients, self.variable)

    def __truediv__(self, other):
        if self.variable != other.variable:
            raise ValueError("Variables must be the same")

        return RationalPolynomial(self, other)

    @staticmethod
    def from_string(poly_string, variable="x"):

        # remove white space
        poly_string = poly_string.replace(" ", "")

        # replace - with +-
        poly_string = poly_string.replace("-","+-")

        # break up into terms
        terms = poly_string.split("+")

        # remove empty elements
        terms_new = []
        for term in terms:
            if term != "":
                terms_new.append(term)
        terms = terms_new

        order, coefficients = Polynomial._process_terms(terms, variable)
        return Polynomial(order, coefficients, variable)

    @staticmethod
    def _process_terms(terms, variable):

        degrees = []
        coeff_list = []

        for term in terms:
            t_list = term.split("*")
            if len(t_list) == 1:
                # Only a single term
                t = t_list[0]
                if not variable in t:
                    # constant
                    degree = 0
                    coeff = int(t)
                else:
                    # coeff is +- 1
                    degree = Polynomial._extract_degree(t)
                    if t[0] == "-":
                        coeff = -1
                    else:
                        coeff = 1
            elif len(t_list) == 2:
                # two terms
                if variable in t_list[0]:
                    t_var = t_list[0]
                    t_coeff = t_list[1]
                    if t_var[0] == "-":
                        # add minus to coeff
                        t_coeff = "-" + t_coeff
                else:
                    t_var = t_list[1]
                    t_coeff = t_list[0]
                degree = Polynomial._extract_degree(t_var)
                coeff = int(t_coeff)

            degrees.append(degree)
            coeff_list.append(coeff)

        order = max(degrees)
        coefficients = np.zeros(order+1, dtype=int)

        for deg, coeff in zip(degrees, coeff_list):
            coefficients[deg] = coeff

        return order, coefficients

    @staticmethod
    def _extract_degree(t):
        carrot = t.find("^")
        if carrot == -1:
            degree = 1
        else:
            degree = int(t[carrot+1:])
        return degree


class RationalPolynomial(FractionBase):

    def __init__(self, numerator, denominator):
        self.fraction_class = RationalPolynomial
        if numerator.variable != denominator.variable:
            raise ValueError("Variables of numerator denominator must be the same")
        super().__init__(numerator, denominator)

    def _reduce(self):
        gcd, numerator, denominator = cofactors(self.numerator.__repr__(), self.denominator.__repr__())

        # convert to strings, switch from ** to ^
        numerator_str = str(numerator).replace("**","^")
        denominator_str = str(denominator).replace("**","^")

        self.numerator = Polynomial.from_string(numerator_str, self.numerator.variable)
        self.denominator = Polynomial.from_string(denominator_str, self.denominator.variable)

        # there is an ambiguity associated with multiplying numerator and denominator by -1
        # fix this by making coefficient of highest order polynomial in the numerator positive
        if self.numerator.coefficients[-1] < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator

    def __repr__(self):
        if self.numerator == Polynomial.from_string("0"):
            return "0"
        elif self.denominator == Polynomial.from_string("1"):
            return self.numerator.__repr__()
        else:
            return "( " + self.numerator.__repr__() + " ) / ( " + self.denominator.__repr__() + " )"

    @staticmethod
    def from_string(string):
        numerator, denominator = string.split('/')
        # delete ( and )
        numerator = numerator.replace("(", "")
        numerator = numerator.replace(")", "")
        denominator = denominator.replace("(", "")
        denominator = denominator.replace(")", "")

        numerator = Polynomial.from_string(numerator)
        denominator = Polynomial.from_string(denominator)
        return RationalPolynomial(numerator, denominator)

