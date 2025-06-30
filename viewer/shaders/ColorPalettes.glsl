
const int sdfPaletteNumColors = 7;
const vec3 sdfPalette[7] = vec3[7](
	vec3(0.0f, 0.0f, 1.0f), 
	vec3(0.0f, 0.5f, 1.0f), 
	vec3(0.0f, 1.0f, 1.0f), 
	vec3(1.0f, 1.0f, 1.0f), 
	vec3(1.0f, 1.0f, 0.0f), 
	vec3(1.0f, 0.5f, 0.0f), 
	vec3(1.0f, 0.0f, 0.0f)
);

const int viridisPaletteNumColors = 6;
const vec3 viridisPalette[6] = vec3[6](
    vec3(68./255., 1./255., 84./255.),
    vec3(65./255., 68./255., 135./255.),
    vec3(42./255., 120./255., 142./255.),
    vec3(34./255., 163./255., 132./255.),
    vec3(122./255., 209./255., 81./255.),
    vec3(253./255., 231./255., 37./255.)
);

const int magmaPaletteNumColors = 9;
const vec3 magmaPalette[9] = vec3[9](
    vec3(0./255., 0./255., 5./255.),
    vec3(26./255., 10./255., 64./255.), 
    vec3(75./255., 0./255., 108./255.), 
    vec3(132./255., 24./255., 109./255.), 
    vec3(198./255., 43./255., 91./255.), 
    vec3(243./255., 95./255., 74./255.), 
    vec3(252./255., 172./255., 109./255.), 
    vec3(251./255., 255./255., 178./255.),
    vec3(1.)
);

vec3 getSdfPaletteColor(float val)
{
    float index = clamp(val * (sdfPaletteNumColors-1), 0.0, float(sdfPaletteNumColors-1) - 0.01);
    return mix(sdfPalette[int(index)], sdfPalette[int(index)+1], fract(index));
}

vec3 getViridisPaletteColor(float val)
{
    float index = clamp(val * (viridisPaletteNumColors-1), 0.0, float(viridisPaletteNumColors-1) - 0.01);
    return mix(viridisPalette[int(index)], viridisPalette[int(index)+1], fract(index));
}

vec3 getMagmaPaletteColor(float val)
{
    float index = clamp(val * (magmaPaletteNumColors-1), 0.0, float(magmaPaletteNumColors-1) - 0.01);
    return mix(magmaPalette[int(index)], magmaPalette[int(index)+1], fract(index));
}