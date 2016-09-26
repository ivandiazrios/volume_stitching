#ifndef MAT4X4_HPP
#define MAT4X4_HPP

#include <fstream>
#include <math.h>
#include <iostream>
#include <iomanip>

template <class T>
class mat4x4 {
public:
    T m[4][4];
    
    mat4x4() : m() {};
    
    mat4x4(const char *input_file) {
        std::ifstream f(input_file);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                f >> m[i][j];
    }
    
    mat4x4(T arr[4][4]) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m[i][j] = arr[i][j];
            }
        }
    }
    
    mat4x4 inverse() {
        T inv[16], det;
        int i;
        
        T *m = &this->m[0][0];
        
        inv[0] = m[5]  * m[10] * m[15] -
        m[5]  * m[11] * m[14] -
        m[9]  * m[6]  * m[15] +
        m[9]  * m[7]  * m[14] +
        m[13] * m[6]  * m[11] -
        m[13] * m[7]  * m[10];
        
        inv[4] = -m[4]  * m[10] * m[15] +
        m[4]  * m[11] * m[14] +
        m[8]  * m[6]  * m[15] -
        m[8]  * m[7]  * m[14] -
        m[12] * m[6]  * m[11] +
        m[12] * m[7]  * m[10];
        
        inv[8] = m[4]  * m[9] * m[15] -
        m[4]  * m[11] * m[13] -
        m[8]  * m[5] * m[15] +
        m[8]  * m[7] * m[13] +
        m[12] * m[5] * m[11] -
        m[12] * m[7] * m[9];
        
        inv[12] = -m[4]  * m[9] * m[14] +
        m[4]  * m[10] * m[13] +
        m[8]  * m[5] * m[14] -
        m[8]  * m[6] * m[13] -
        m[12] * m[5] * m[10] +
        m[12] * m[6] * m[9];
        
        inv[1] = -m[1]  * m[10] * m[15] +
        m[1]  * m[11] * m[14] +
        m[9]  * m[2] * m[15] -
        m[9]  * m[3] * m[14] -
        m[13] * m[2] * m[11] +
        m[13] * m[3] * m[10];
        
        inv[5] = m[0]  * m[10] * m[15] -
        m[0]  * m[11] * m[14] -
        m[8]  * m[2] * m[15] +
        m[8]  * m[3] * m[14] +
        m[12] * m[2] * m[11] -
        m[12] * m[3] * m[10];
        
        inv[9] = -m[0]  * m[9] * m[15] +
        m[0]  * m[11] * m[13] +
        m[8]  * m[1] * m[15] -
        m[8]  * m[3] * m[13] -
        m[12] * m[1] * m[11] +
        m[12] * m[3] * m[9];
        
        inv[13] = m[0]  * m[9] * m[14] -
        m[0]  * m[10] * m[13] -
        m[8]  * m[1] * m[14] +
        m[8]  * m[2] * m[13] +
        m[12] * m[1] * m[10] -
        m[12] * m[2] * m[9];
        
        inv[2] = m[1]  * m[6] * m[15] -
        m[1]  * m[7] * m[14] -
        m[5]  * m[2] * m[15] +
        m[5]  * m[3] * m[14] +
        m[13] * m[2] * m[7] -
        m[13] * m[3] * m[6];
        
        inv[6] = -m[0]  * m[6] * m[15] +
        m[0]  * m[7] * m[14] +
        m[4]  * m[2] * m[15] -
        m[4]  * m[3] * m[14] -
        m[12] * m[2] * m[7] +
        m[12] * m[3] * m[6];
        
        inv[10] = m[0]  * m[5] * m[15] -
        m[0]  * m[7] * m[13] -
        m[4]  * m[1] * m[15] +
        m[4]  * m[3] * m[13] +
        m[12] * m[1] * m[7] -
        m[12] * m[3] * m[5];
        
        inv[14] = -m[0]  * m[5] * m[14] +
        m[0]  * m[6] * m[13] +
        m[4]  * m[1] * m[14] -
        m[4]  * m[2] * m[13] -
        m[12] * m[1] * m[6] +
        m[12] * m[2] * m[5];
        
        inv[3] = -m[1] * m[6] * m[11] +
        m[1] * m[7] * m[10] +
        m[5] * m[2] * m[11] -
        m[5] * m[3] * m[10] -
        m[9] * m[2] * m[7] +
        m[9] * m[3] * m[6];
        
        inv[7] = m[0] * m[6] * m[11] -
        m[0] * m[7] * m[10] -
        m[4] * m[2] * m[11] +
        m[4] * m[3] * m[10] +
        m[8] * m[2] * m[7] -
        m[8] * m[3] * m[6];
        
        inv[11] = -m[0] * m[5] * m[11] +
        m[0] * m[7] * m[9] +
        m[4] * m[1] * m[11] -
        m[4] * m[3] * m[9] -
        m[8] * m[1] * m[7] +
        m[8] * m[3] * m[5];
        
        inv[15] = m[0] * m[5] * m[10] -
        m[0] * m[6] * m[9] -
        m[4] * m[1] * m[10] +
        m[4] * m[2] * m[9] +
        m[8] * m[1] * m[6] -
        m[8] * m[2] * m[5];
        
        det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        
        // Ideally should check if det is 0
        det = T(1.0) / det;
        
        T result[16];
        
        for (i = 0; i < 16; i++)
            result[i] = inv[i] * det;
        
        return mat4x4((T (*)[4]) result);
    }
    
    void write_to_file(const char *file) {
        
        std::ofstream output(file);
        for (int i=0;i<4;i++) {
            for (int j=0;j<4;j++) {
                output << m[i][j] << " ";
            }
            output << "\n";
        }
    }
    
    mat4x4 operator*(const mat4x4 &m2) const {
        
        mat4x4 product;
        
        const T (*a)[4] = this->m;
        const T (*b)[4] = m2.m;
        
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                for (int inner = 0; inner < 4; inner++) {
                    product.m[row][col] += a[row][inner] * b[inner][col];
                }
            }
        }
        
        return product;
    }
    
    mat4x4 translate(T x, T y, T z) {
        
        mat4x4 copy(this->m);
        
        copy.m[0][3] += x;
        copy.m[1][3] += y;
        copy.m[2][3] += z;
        
        return copy;
    }
    
    mat4x4 rotate_x(float x_angle) {
        
        mat4x4 copy(this->m);
        
        float c = cos(x_angle);
        float s = sin(x_angle);
        
        copy.m[1][0] = c * m[1][0] - s * m[2][0];
        copy.m[1][1] = c * m[1][1] - s * m[2][1];
        copy.m[1][2] = c * m[1][2] - s * m[2][2];
        copy.m[1][3] = c * m[1][3] - s * m[2][3];
        
        copy.m[2][0] = s * m[1][0] + c * m[2][0];
        copy.m[2][1] = s * m[1][1] + c * m[2][1];
        copy.m[2][2] = s * m[1][2] + c * m[2][2];
        copy.m[2][3] = s * m[1][3] + c * m[2][3];
        
        return copy;
    }
    
    mat4x4 rotate_y(float y_angle) {
        
        mat4x4 copy(this->m);
        
        float c = cos(y_angle);
        float s = sin(y_angle);
        
        copy.m[0][0] = c * m[0][0] + s * m[2][0];
        copy.m[0][1] = c * m[0][1] + s * m[2][1];
        copy.m[0][2] = c * m[0][2] + s * m[2][2];
        copy.m[0][3] = c * m[0][3] + s * m[2][3];
        
        copy.m[2][0] = -s * m[0][0] + c * m[2][0];
        copy.m[2][1] = -s * m[0][1] + c * m[2][1];
        copy.m[2][2] = -s * m[0][2] + c * m[2][2];
        copy.m[2][3] = -s * m[0][3] + c * m[2][3];
        
        return copy;
        
    }
    
    mat4x4 rotate_z(float z_angle) {
        
        mat4x4 copy(this->m);
        
        float c = cos(z_angle);
        float s = sin(z_angle);
        
        copy.m[0][0] = c * m[0][0] - s * m[1][0];
        copy.m[0][1] = c * m[0][1] - s * m[1][1];
        copy.m[0][2] = c * m[0][2] - s * m[1][2];
        copy.m[0][3] = c * m[0][3] - s * m[1][3];
        
        copy.m[1][0] = s * m[0][0] + c * m[1][0];
        copy.m[1][1] = s * m[0][1] + c * m[1][1];
        copy.m[1][2] = s * m[0][2] + c * m[1][2];
        copy.m[1][3] = s * m[0][3] + c * m[1][3];
        
        return copy;
        
    }
    
    static mat4x4 bryant_to_matrix(float angle_x, float angle_y, float angle_z) {
        
        // Produce rotation matrix from bryant angles
        // http://www.u.arizona.edu/~pen/ame553/Notes/Lesson%2008-B.pdf
        
        float c1 = cos(angle_x), c2 = cos(angle_y), c3 = cos(angle_z);
        float s1 = sin(angle_x), s2 = sin(angle_y), s3 = sin(angle_z);
        
        T arr[4][4] = {
            {c2*c3,          -c2*s3,         s2,     0},
            {c1*s3+s1*s2*c3, c1*c3-s1*s2*s3, -s1*c2, 0},
            {s1*s3-c1*s2*c3, s1*c3+c1*s2*s3, c1*c2,  0},
            {0,              0,              0,      1}
        };
        
        return mat4x4(arr);
    }
    
    void pretty_print() {
        std::cout << std::fixed << std::setprecision(5);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << m[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    T sum_squared_difference(mat4x4 &mat) {
        
        T ssd = 0;
        
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                
                T diff = m[i][j] - mat.m[i][j];
                ssd += diff * diff;
                
            }
        
        return ssd;
    }
    
    static mat4x4 get_identity_matrix() {
        mat4x4 mat;
        mat.m[0][0] = T(1);
        mat.m[1][1] = T(1);
        mat.m[2][2] = T(1);
        mat.m[3][3] = T(1);
        return mat;
    }
};


#endif