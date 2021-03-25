#include <unittest/unittest.h>

#include <thrust/vector_reference.h>

void TestVectorReferenceBasic(void)
{
  thrust::vector_reference<int> v;
  ASSERT_EQUAL(v.empty(),true);

  const int arraySize = 7;
  int data[arraySize] = {10,6,2,50,7,8,5};
  v = thrust::vector_reference<int>(data, arraySize);

  ASSERT_EQUAL(v.empty(),false);
  ASSERT_EQUAL(v.size(), 7u);

  ASSERT_EQUAL(v[0], 10);
  ASSERT_EQUAL(v[3], 50);
  ASSERT_EQUAL(v[6], 5);
  ASSERT_EQUAL(v.at(1), 6);
  ASSERT_EQUAL(v.at(4), 7);
  ASSERT_EQUAL(v.at(5), 8);

  v[0] = 13;
  v[3] = 17;
  v[6] = 42;
  v.at(1) = 64; 
  v.at(4) = 845;
  v.at(5) = 0;

  ASSERT_EQUAL(v.front(), 13);
  ASSERT_EQUAL(v.back(), 42);

  ASSERT_EQUAL(data[3], 17);
  ASSERT_EQUAL(data[4], 845);
  ASSERT_EQUAL(data[5], 0);

  ASSERT_EQUAL(data, v.data());

  thrust::vector_reference<int> v2;
  swap(v,v2);
  ASSERT_EQUAL(v.empty(), true);
  ASSERT_EQUAL(v2.empty(), false);
  ASSERT_EQUAL(v2[0], 13);
}
DECLARE_UNITTEST(TestVectorReferenceBasic);

void TestVectorReferenceOutOfBounds(void)
{
  const int arraySize = 7;
  int data[arraySize] = {10,6,2,50,7,8,5};
  thrust::vector_reference<int> v(data, arraySize);
  ASSERT_THROWS(v.at(25)=10, std::out_of_range);

  thrust::vector_reference<int> c(data, arraySize);
  ASSERT_THROWS(c.at(25), std::out_of_range);
}
DECLARE_UNITTEST(TestVectorReferenceOutOfBounds);


void TestVectorReferenceSTDvec(void)
{
  std::vector<float> vec = {10.0f,6.0f,2.0f,50.0f,7.0f,8.0f,5.0f};
  
  auto vref = thrust::make_vector_reference(vec);
  ASSERT_EQUAL(vec.size(), vref.size());
  ASSERT_EQUAL(vec[1], vref[1]);

  vref[5] = 0.25f;
  ASSERT_EQUAL(vec[5], vref[5]);
}
DECLARE_UNITTEST(TestVectorReferenceSTDvec);

template <class Vector>
void TestVectorReferenceFromDifferentVectors(void)
{
  std::vector<int> stdvec = {10,6,2,50,7,8,5};
  
  Vector vec(stdvec);

  auto vref = thrust::make_vector_reference(vec);
  ASSERT_EQUAL(vec.size(), vref.size());
  ASSERT_EQUAL(vec[1], vref[1]);

  vref[5] = 25;
  ASSERT_EQUAL(vec[5], vref[5]);
}
DECLARE_VECTOR_UNITTEST(TestVectorReferenceFromDifferentVectors);


template <class Vector>
void TestVectorReferenceIterators(void)
{
  std::vector<int> stdvec = {10,6,2,50,7,8,5};
  
  Vector vec(stdvec);

  auto vref = thrust::make_vector_reference(vec);
  ASSERT_EQUAL(vec.size(), vref.size());
  ASSERT_EQUAL(vec[1], vref[1]);

  vref[5] = 25;
  ASSERT_EQUAL(vec[5], vref[5]);
}
DECLARE_VECTOR_UNITTEST(TestVectorReferenceIterators);


__global__ void testVectorRefOnDevice(thrust::vector_reference<int> in, thrust::vector_reference<int> out)
{
  if(!in.size() == out.size()) {
    printf("different sizes");
    asm("trap;");
  }

  for(int i = blockIdx.x*blockDim.x+threadIdx.x; i < in.size(); i += blockDim.x*gridDim.x)
  {
    out[i] = in[i];
  }
}

void TestVectorReferenceOnDevice(void)
{
  std::vector<int> stdvec = {10,6,2,50,7,8,5,12,42,10};
  thrust::device_vector<int> vin(stdvec);
  thrust::device_vector<int> vout(vin.size());

  testVectorRefOnDevice<<< 1, 1>>>( thrust::make_device_vector_reference(vin),
                                     thrust::make_device_vector_reference(vout));

  ASSERT_EQUAL(vin[1], vout[1]);
  ASSERT_EQUAL(vin[5], vout[5]);
  ASSERT_EQUAL(vin[8], vout[8]);
}
DECLARE_UNITTEST(TestVectorReferenceOnDevice);

void TestVectorReferenceToConst(void)
{
  const std::vector<float> cvec = {10.0f,6.0f,2.0f,50.0f,7.0f,8.0f,5.0f};
  std::vector<float> vec = {10.0f,6.0f,2.0f,50.0f,7.0f,8.0f,5.0f};

  auto vref = thrust::make_vector_reference(cvec);
  ASSERT_EQUAL(cvec.size(), vref.size());
  ASSERT_EQUAL(cvec[1], vref[1]);
  ASSERT_EQUAL(cvec[4], vref[4]);
  ASSERT_EQUAL(cvec[6], vref[6]);
  ASSERT_EQUAL(cvec.at(2), vref.at(2));
  ASSERT_EQUAL(cvec.at(5), vref.at(5));

  thrust::vector_reference<const float> vref2 = thrust::make_vector_reference(vec);
  ASSERT_EQUAL(vec.size(), vref2.size());
  ASSERT_EQUAL(vec[1], vref2[1]);
  ASSERT_EQUAL(vec[4], vref2[4]);
  ASSERT_EQUAL(vec[6], vref2[6]);
  ASSERT_EQUAL(vec.at(2), vref2.at(2));
  ASSERT_EQUAL(vec.at(5), vref2.at(5));
  
}
DECLARE_UNITTEST(TestVectorReferenceToConst);