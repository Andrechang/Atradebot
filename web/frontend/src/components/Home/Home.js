import React from 'react';
import Slider from './Slider';

export default function Home() {
  return (
    <div>
      <Slider/>
      <section className="pt-80 pb-40 pl-20 pr-20" id="aboutus">
    <div className="font__family-montserrat font__weight-bold font__size-48 line__height-56 mb-10 text-center mt-0">About Us </div>
    <div>
        <p className="font__family-open-sans font__size-18 line__height-28 mt-30 pb-0 text-center maxw-970">
            Algofox is India’s most promising algo trading platform for retail traders and investors in India.
            Made with a simple vision of making algo trading accessible to every trader/investor at a very economical price.
            The platform aims at providing retail traders with speed and precision in order placement across various platforms and brokers and making their trading/investing journey more smoother and profitable.
        </p>
    </div>
    <div className="font__family-montserrat font__weight-bold font__size-48 line__height-56 mb-10 text-center mt-50">Our Mission </div>
    <div>
        <p className="font__family-open-sans font__size-18 line__height-28 mt-30 pb-md-30 text-center maxw-970">
            To provide fast and reliable trading platforms that seamlessly connect with your desired broker and help you place trades systematically.
            We want to empower every retail trader with the power of algorithms and automation to make their trading journey smoother and profitable.
        </p>
    </div>

</section>
    </div>
  )
}
