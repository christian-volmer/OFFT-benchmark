
//          Copyright Christian Volmer 2023.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

// #define USE_MKL
// #define USE_FFTW

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <windows.h>

#include <offt.h>
#include <offt/math.h>

#ifdef USE_FFTW
#include <fftw3.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

double constexpr Pi = 3.1415926535897932385;

class FftData {
	std::size_t const mSize;
	double *mIn, *mOut;
	std::size_t mBin;
	double mPhase;

public:

	FftData(std::size_t size) :
		mSize(size),
		mIn(),
		mOut(),
		mBin(size / 3),
		mPhase(1.0)
	{
#ifdef USE_FFTW
		mIn = (double *)fftw_malloc(sizeof(fftw_complex) * size);
		mOut = (double *)fftw_malloc(sizeof(fftw_complex) * size);
#else
		mIn = (double *)malloc(sizeof(std::complex<double>) * size);
		mOut = (double *)malloc(sizeof(std::complex<double>) * size);
#endif

		for (std::size_t i = 0; i < size; ++i) {
			std::complex<double> p = std::exp(std::complex<double>(0.0, 2 * Pi * i * mBin / size + mPhase));

			mIn[2 * i + 0] = p.real();
			mIn[2 * i + 1] = p.imag();
		}
	}

	~FftData()
	{
#ifdef USE_FFTW
		fftw_free(mIn);
		fftw_free(mOut);
#else
		free(mIn);
		free(mOut);
#endif
	}

	std::size_t GetSize() const
	{
		return mSize;
	}

	template<typename T>
	T const *GetIn() const
	{
		return reinterpret_cast<T const *>(mIn);
	}

	template<typename T>
	T *GetOut() const
	{
		return reinterpret_cast<T *>(mOut);
	}

	void Clear()
	{
		std::fill(mOut, mOut + 2 * mSize, 0.0);
	}

	void Check()
	{
		std::complex<double> expectedBinValue = double(mSize) * std::exp(std::complex<double>(0.0, mPhase));

		std::complex<double> binDifference = expectedBinValue - std::complex<double>(mOut[2 * mBin], mOut[2 * mBin + 1]);

		// The error at the single non-zero frequency bin
		double binError = std::abs(binDifference) / std::abs(expectedBinValue);

		// The deviation from zero at all other bins
		double zeroError = 0.0;
		for (std::size_t i = 0; i < mSize; ++i) {

			if (i == mBin)
				continue;

			zeroError += mOut[2 * i] * mOut[2 * i] + mOut[2 * i + 1] * mOut[2 * i + 1];
		}

		zeroError = std::sqrt(zeroError) / std::abs(expectedBinValue);

		if (binError > 1e-8 || zeroError > 1e-8) {

			std::cout << "\n\n -- Error in FFT computation --\n"
					  << std::defaultfloat << std::setprecision(3);
			std::cout << "  binError  = " << binError << ", ";
			std::cout << "  zeroError = " << zeroError << "\n\n";

			// throw std::runtime_error("Error in FFT computation.");
		}
	}
};

class Runner {
private:

	FftData &mFftData;

public:

	Runner(FftData &fftData) :
		mFftData(fftData) {};

	FftData &GetFftData() const
	{
		return mFftData;
	}
};

class OfftRunner : public Runner {

private:

	offt::Fourier<> mFourier;
	std::complex<double> const *mIn;
	std::complex<double> *mOut;

public:

	OfftRunner(FftData &fftData) :
		Runner(fftData),
		mFourier(fftData.GetSize(), offt::FourierParameters::FFTW),
		mIn(fftData.GetIn<std::complex<double>>()),
		mOut(fftData.GetOut<std::complex<double>>())
	{
		mFourier.EnsureTemp();
	}

	void Run()
	{
		mFourier.Transform(mOut, 1, mIn, 1);
	}
};

#ifdef USE_FFTW

class FftwRunner : public Runner {

private:

	fftw_plan mPlan;
	fftw_complex const *mIn;
	fftw_complex *mOut;

public:

	FftwRunner(FftData &fftData, unsigned flags) :
		Runner(fftData),
		mPlan(),
		mIn(fftData.GetIn<fftw_complex>()),
		mOut(fftData.GetOut<fftw_complex>())
	{
		mPlan = fftw_plan_dft_1d((int)fftData.GetSize(), const_cast<fftw_complex *>(mIn), mOut, FFTW_FORWARD, flags);
	}

	~FftwRunner()
	{
		fftw_destroy_plan(mPlan);
	}

	void Run()
	{
		fftw_execute(mPlan);
	}
};

#endif // USE_FFTW

#ifdef USE_MKL

class MklRunner : public Runner {
private:

	DFTI_DESCRIPTOR_HANDLE mDesc;
	std::complex<double> const *mIn;
	std::complex<double> *mOut;

public:

	MklRunner(FftData &fftData, int cores) :
		Runner(fftData),
		mDesc(),
		mIn(fftData.GetIn<std::complex<double>>()),
		mOut(fftData.GetOut<std::complex<double>>())
	{
		DftiCreateDescriptor(&mDesc, DFTI_DOUBLE, DFTI_COMPLEX, 1, fftData.GetSize());
		DftiSetValue(mDesc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
		DftiSetValue(mDesc, DFTI_THREAD_LIMIT, cores);
		DftiCommitDescriptor(mDesc);
	}

	~MklRunner()
	{
		DftiFreeDescriptor(&mDesc);
	}

	void Run()
	{
		DftiComputeForward(mDesc, const_cast<std::complex<double> *>(mIn), mOut);
	}
};

#endif // USE_MKL

static int const gMinimumCount = 4;
static double const gMinimumDuration = 1;

template<typename T>
double Timer(T &runner)
{
	runner.GetFftData().Clear();
	runner.Run();
	runner.GetFftData().Check();

	double duration = 0;
	size_t count = 0;

	auto startTime = std::chrono::steady_clock::now();

	do {

		runner.Run();
		duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
		++count;

	} while (count < gMinimumCount || duration < gMinimumDuration);

	double size(runner.GetFftData().GetSize());

	return 1e9 * duration / count / (size * std::log(size));
}

void PrintFactors(std::size_t size)
{
	std::vector<std::ptrdiff_t> factors;

	offt::math::FactorInteger(size, [&](std::ptrdiff_t factor) { factors.push_back(factor); });
	std::sort(factors.begin(), factors.end());

	for (auto f = factors.cbegin(); f != factors.cend();) {

		int multiplicity = 1;
		ptrdiff_t factor = *f;
		for (++f; *f == factor && f != factors.cend(); ++f)
			++multiplicity;

		std::cout << factor;

		if (multiplicity > 1)
			std::cout << "<sup>" << multiplicity << "</sup>";

		if (f != factors.cend())
			std::cout << "&thinsp;&middot;&thinsp;";
	}
}

void RunList(std::vector<size_t> const &sizes)
{
	std::cout << "|    Size     |    OFFT    |";
#ifdef USE_FFTW
	std::cout << " FFTW w/ SIMD | FFTW w/o SIMD |";
#endif
#ifdef USE_MKL
	std::cout << " MKL 1-core | MKL 2-core |";
#endif
	std::cout << " Factors\n";

	std::cout << "|-------------|------------|";
#ifdef USE_FFTW
	std::cout << "--------------|---------------|";
#endif
#ifdef USE_MKL
	std::cout << "------------|------------|";
#endif

	std::cout << "-----------\n";

	for (auto size : sizes) {

		std::cout << std::setprecision(1);
		std::cout << std::fixed;

		FftData data(size);

		OfftRunner offtRunner(data);
#ifdef USE_FFTW
		FftwRunner fftwRunner(data, FFTW_ESTIMATE);
		FftwRunner fftwNoSimdRunner(data, FFTW_ESTIMATE | FFTW_NO_SIMD);
#endif
#ifdef USE_MKL
		MklRunner mklStRunner(data, 1);
		MklRunner mklMtRunner(data, 2);
#endif

		std::cout << "| ";

		std::cout << std::setw(11);
		std::cout << size;

		std::cout << " | ";

		std::cout << std::setw(10);
		std::cout << Timer(offtRunner);

#ifdef USE_FFTW

		std::cout << " | ";

		std::cout << std::setw(12);
		std::cout << Timer(fftwRunner);

		std::cout << " | ";

		std::cout << std::setw(13);
		std::cout << Timer(fftwNoSimdRunner);

#endif

#ifdef USE_MKL

		std::cout << " | ";

		std::cout << std::setw(10);
		std::cout << Timer(mklStRunner);

		std::cout << " | ";

		std::cout << std::setw(10);
		std::cout << Timer(mklMtRunner);

#endif

		std::cout << " | ";

		PrintFactors(size);

		std::cout << " \n";
	}
}

int main()
{
	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

	{
		std::cout << "### Powers of two\n\n";

		std::vector<size_t> powersOfTwo;
		for (std::size_t i = 10; i <= 24; ++i)
			powersOfTwo.push_back(std::pow(2u, i));

		RunList(powersOfTwo);
	}

	std::cout << "\n";

	{
		std::cout << "### Powers of ten\n\n";

		std::vector<size_t> powersOfTen;
		for (std::size_t i = 2; i <= 7; ++i)
			powersOfTen.push_back(std::pow(10u, i));

		RunList(powersOfTen);
	}

	std::cout << "\n";

	{
		std::cout << "### Non-powers of two\n\n";

		std::vector<size_t> nonPowersOfTwo;
		for (std::size_t i = 10; i <= 24; ++i) {

			nonPowersOfTwo.push_back(std::pow(2u, i) - 1);
			nonPowersOfTwo.push_back(std::pow(2u, i) + 1);
		}

		RunList(nonPowersOfTwo);
	}
}