
float erf(const float x)
{
    //  Float version:
	const float sign_x = sign(x);
	const float t = 1.0/(1.0 + 0.47047*abs(x));
	const float result = 1.0 - t*(0.3480242 + t*(-0.0958798 + t*0.7478556))*
		exp(-(x*x));
	return result * sign_x;
}