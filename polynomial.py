# -*- coding: utf-8 -*-
"""
ES_APPM 446 HW 1
Due Date: 9/30/22

@author: Nicolas Guerra
"""
import numpy as np
from sympy import simplify
from sympy import expand


class Polynomial:
    def __init__(self, order, coefficients):
        self.order = order
        self.coefficients = coefficients
        self._remove()  # removes padded zeros from coef. array

    @staticmethod
    def from_string(string):
        string = string.replace(" ", "")
        split_minus = string.split('-')

        monoms = []
        # Check there are any negative monomials
        if len(split_minus) > 1:
            # Check if initial element was negative
            # If so, remove empty string
            empty = ''
            if empty in split_minus:
                split_minus.remove('')
                start = 0
            else:
                start = 1  # The first element is not negative so skip it
            # Add first positive monomial
            if start == 1:
                for element_plus in split_minus[0].split('+'):
                    monoms.append(element_plus)
            # Add the other monomials
            for i in range(start, len(split_minus)):
                concat = '-'+split_minus[i]
                for element_plus in concat.split('+'):
                    monoms.append(element_plus)
        else:
            monoms = split_minus[0].split('+')

        # Break up monomial
        ordering = []
        coefficients_list = []
        for monomial in monoms:
            # Check coefficient is alone
            if monomial.find('x') != -1:
                # Find out power
                if monomial.find('^') != -1:
                    idx = monomial.find('^')
                    ordering.append( int(monomial[idx+1:]) )
                else:
                    ordering.append(1)
                # Find coefficient
                if monomial.find('*') != -1:
                    idx = monomial.find('*')
                    coefficients_list.append( int(monomial[:idx]) )
                else:
                    if '-' in monomial:
                        coefficients_list.append(-1)
                    else:
                        coefficients_list.append(1)
            else:
                coefficients_list.append( int(monomial) )
                ordering.append(0)

        # Set up to make new object
        order = max(ordering)
        coefficients = np.zeros(order+1,dtype=int)
        counter = 0
        for index in ordering:
            coefficients[index] = coefficients_list[counter]
            counter += 1
            
        return Polynomial(order, coefficients)

    # For printing polynomial
    def __repr__(self):
        string = str(self.coefficients[0])
        for i in range(1,self.order+1):
            if self.coefficients[i] > 0:
                string += '+'+str(self.coefficients[i])+'*x^'+str(i)
            elif self.coefficients[i] < 0:
                string += str(self.coefficients[i])+'*x^'+str(i)
        return string
    
    # Addition
    def __add__(self,other):
        if self.order <= other.order:
            coef_temp = np.copy(other.coefficients)
            order_temp = other.order
            for i in range(self.order+1):
                coef_temp[i] += self.coefficients[i]
        else:
            coef_temp = np.copy(self.coefficients)
            order_temp = self.order
            for i in range(other.order+1):
                coef_temp[i] += other.coefficients[i]
        return Polynomial(order_temp, coef_temp)
    
    # Negative polynomial
    def __neg__(self):
        coef_temp = -1*self.coefficients
        return Polynomial(self.order, coef_temp)
    
    # Subtraction
    def __sub__(self, other):
        return self + (-other)
    
    # Multiplication
    def __mul__(self, other):
        # We have to flip the coefficients for np.polymul()
        coefficients_flipped = np.polymul( np.flip(self.coefficients), np.flip(other.coefficients) )
        coefficients = np.flip(coefficients_flipped)
        order = len(coefficients)-1
        return Polynomial(order, coefficients)
    
    # Equality
    def __eq__(self, other):
        if list(self.coefficients) == list(other.coefficients):
            return True
        else:
            return False
        
    # Division
    def __truediv__(self,other):
        return RationalPolynomial(self,other)
        
    # Remove padded zeros 
    def _remove(self):
        coef_list = list(self.coefficients)
        while coef_list[-1] == 0:
            coef_list.pop(-1)
        self.coefficients = np.array(coef_list)
        self.order = len(self.coefficients)-1
        

class RationalPolynomial(Polynomial):
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self._simplify()
        
    @staticmethod
    def from_string(string):
        numerator_string = string.split('/')[0]
        denominator_string = string.split('/')[1]
        
        # Get rid of parenthesis
        numerator_string = numerator_string.replace('(','')
        numerator_string = numerator_string.replace(')','')
        denominator_string = denominator_string.replace('(','')
        denominator_string = denominator_string.replace(')','')
        
        numerator = Polynomial.from_string(numerator_string)
        denominator = Polynomial.from_string(denominator_string)
        return RationalPolynomial(numerator, denominator)
    
    def __repr__(self):
        string = '('+str(self.numerator)+')/('+str(self.denominator)+')'
        return string
        
    def _simplify(self):
        string = str(simplify(str(self))).replace('**','^')
        
        numerator_string = string.split('/')[0]
        # expands out top and bottom
        numerator_string = str(expand(numerator_string)).replace('**','^')
        # try if there is a denominator
        try:
            denominator_string = string.split('/')[1]
            denominator_string = str(expand(denominator_string)).replace('**','^')
        except IndexError:
            denominator_string = str(1)
        
        # Get rid of parenthesis
        numerator_string = numerator_string.replace('(','')
        numerator_string = numerator_string.replace(')','')
        denominator_string = denominator_string.replace('(','')
        denominator_string = denominator_string.replace(')','')
        
        self.numerator = Polynomial.from_string(numerator_string)
        self.denominator = Polynomial.from_string(denominator_string)
        
    # Addition
    def __add__(self, other):
        numerator = ( self.numerator * other.denominator + other.numerator *  self.denominator)
        denominator = self.denominator * other.denominator
        return RationalPolynomial(numerator, denominator)

    # Negative
    def __neg__(self):
        numerator = -self.numerator
        return RationalPolynomial(numerator, self.denominator)

    # Subtraction
    def __sub__(self, other):
        return self + (-other)

    # Multiplication:
    def __mul__(self, other):
        return RationalPolynomial(self.numerator*other.numerator,
                                  self.denominator*other.denominator)

    # Division
    def __truediv__(self, other):
        return RationalPolynomial(self.numerator*other.denominator,
                                  self.denominator*other.numerator)

    # Equality
    def __eq__(self, other):
        if self.numerator == other.numerator:
            if self.denominator == other.denominator:
                return True
        return False
