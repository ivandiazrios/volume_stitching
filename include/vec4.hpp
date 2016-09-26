#ifndef VEC4_HPP
#define VEC4_HPP

template <class T>
class vec4 {
public:
    T a1,a2,a3,a4;
    
    vec4() : a1(), a2(), a3(), a4() {};
    
#ifdef __CUDACC__
    __host__ __device__
#endif
    vec4(T a1, T a2, T a3, T a4) : a1(a1), a2(a2), a3(a3), a4(a4) {};
    
#ifdef __CUDACC__
    __host__ __device__
#endif
    float get_distance(vec4 &v2) {
        float s1 = (a1 - v2.a1);
        float s2 = (a2 - v2.a2);
        float s3 = (a3 - v2.a3);
        return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
    }
};

template<class T>
#ifdef __CUDACC__
__host__ __device__
#endif
vec4<T> operator* (mat4x4<T>& mat, const vec4<T> vec) {
    T a1=mat.m[0][0]*vec.a1 + mat.m[0][1]*vec.a2 + mat.m[0][2]*vec.a3 + mat.m[0][3];
    T a2=mat.m[1][0]*vec.a1 + mat.m[1][1]*vec.a2 + mat.m[1][2]*vec.a3 + mat.m[1][3];
    T a3=mat.m[2][0]*vec.a1 + mat.m[2][1]*vec.a2 + mat.m[2][2]*vec.a3 + mat.m[2][3];
    T a4=mat.m[3][0]*vec.a1 + mat.m[3][1]*vec.a2 + mat.m[3][2]*vec.a3 + mat.m[3][3];
    return vec4<T>(a1, a2, a3, a4);
}

#endif