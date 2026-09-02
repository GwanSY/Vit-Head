#include <iostream>
using namespace std;

const double EPISON = 1e-7;
class Complex
{
public:
    int64_t real;
  	int64_t image;
  	Complex(const Complex& complex) :real{ complex.real }, image{ complex.image } {

  	}
  	Complex(int64_t Real=0, int64_t Image=0) :real{ Real }, image{ Image } {

  	}
  	//TODO
    Complex operator+(const Complex c) {
        return Complex(this->real + c.real, this->image + c.image);
    }
    
    Complex operator-(const Complex c) {
        return Complex(this->real - c.real, this->image - c.image);
    }
    
    Complex operator*(const Complex c) {
        int64_t _real = this->real * c.real - this->image * c.image;
        int64_t _image = this->image * c.real + this->real * c.image;
        return Complex(_real, _image);
    }
    
    // Complex operator/(const Complex c) {
    //     int64_t _real = (this->real * c.real + this->image * c.image) / (c.real * c.real + c.image * c.image);
    //     int64_t _image = (this->image * c.real - this->real * c.image) / (c.real * c.real + c.image * c.image);
    //     return Complex(_real, _image);
    // }
    // friend istream &operator>>(istream &in, Complex &c);
    // friend ostream &operator<<(ostream &out, const Complex &c);
};

// //重载>>
// istream &operator>>(istream &in, Complex &c) {
//     in >> c.real >> c.image;
//     return in;
// }

// //重载<<
// ostream &operator<<(ostream &out, const Complex &c) {
//     out << "(";
//     //判断实部是否为正数或0
//     if (c.real >= EPISON || (c.real < EPISON && c.real > -EPISON)) out.unsetf(std::ios::showpos);
//     out << c.real;
//     out.setf(std::ios::showpos);
//     out << c.image;
//     out << "i)";
//     return out;
// }
